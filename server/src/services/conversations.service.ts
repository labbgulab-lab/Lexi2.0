import dotenv from 'dotenv';
import mongoose from 'mongoose';
import crypto from 'crypto';
import fs from "fs";
import FormData from 'form-data';
import {
    AgentsMode,
    IAgent,
    Message,
    SegmentCloseReason,
    SegmentRecord,
    SegmentsConfig,
    SilentRoundRecord,
    TurnTakingConfig,
    UserAnnotation,
    Audio,
    LLMProvider,
} from '../types';
import { ConversationsModel } from '../models/ConversationsModel';
import { ExplainableModel } from '../models/ExplainableModel';
import { MetadataConversationsModel } from '../models/MetadataConversationsModel';
import { experimentsService } from './experiments.service';
import { usersService } from './users.service';
import { apiKeysService } from './apiKeys.service';
import { createChatCompletion, getProviderFromModel } from './llmProviders.service';
import {
    StoredMsg,
    ModeratorFn,
    assembleHistory,
    buildFirstSentenceDocs,
    createGroupChatCompletion,
    createModeratorCompletion,
    resolveGroupAgents,
    runGroupRound,
} from './groupChat.service';
import {
    resolvePacing,
    isHumanized,
    isHumanizedOneOnOne,
    isSplitEnabled,
    resolveTypingDelay,
    buildPacedSegments,
    NO_OP_PACING,
} from './pacing.service';
import {
    applyPhase,
    chatElapsedMs,
    hasPhases,
    mergeAgentPhases,
    PhaseClock,
    PhaseResolution,
    resolvePhase,
} from './phases.service';
import {
    applyKnowledge,
    ClientSegmentView,
    clientSegmentView,
    composeKnowledge,
    evaluateQuota,
    isFinalSegment,
    renderKnowledgeTemplate,
    isSegmented,
    openSegment,
    quotaForSegment,
    resolveEntry,
    transcriptPositionOf,
    userMessagesInSegment,
} from './segments.service';
import {
    ResolvedThreadControl,
    resolveMemoryWindow,
    resolveScene,
    resolveThreadControl,
} from './threadControl.service';
import { makeRng, mulberry32 } from '../utils/seededRandom';
import { AppError } from '../utils/errors';
import { CONVERSATION_LOCK_MAX_HOLD_MS } from '../utils/conversationLock';
import { CurrentStateModels } from '../models/CurrentStateModels';
import fetch from "node-fetch";
import Meyda from 'meyda';
import ffmpeg from "fluent-ffmpeg";
import ffmpegPath from "@ffmpeg-installer/ffmpeg";
import path from "path";
import os from "os";
import { v4 as uuidv4 } from 'uuid';

ffmpeg.setFfmpegPath(ffmpegPath.path);

dotenv.config();

async function convertToWav(inputBuffer: Buffer): Promise<Buffer> {
    return new Promise((resolve, reject) => {
        const tempInput = path.join(os.tmpdir(), `temp_input_${uuidv4()}.wav`);
        const tempOutput = path.join(os.tmpdir(), `temp_output_${uuidv4()}.wav`);

        try {
            fs.writeFileSync(tempInput, inputBuffer);

            ffmpeg(tempInput)
                .output(tempOutput)
                .audioCodec("pcm_s16le")
                .toFormat("wav")
                .on("end", () => {
                    try {
                        const wavBuffer = fs.readFileSync(tempOutput);
                        resolve(wavBuffer);
                    } catch (err) {
                        reject(err);
                    } finally {
                        try { fs.unlinkSync(tempInput); } catch {}
                        try { fs.unlinkSync(tempOutput); } catch {}
                    }
                })
                .on("error", (err) => {
                    try { fs.unlinkSync(tempInput); } catch {}
                    try { fs.unlinkSync(tempOutput); } catch {}
                    reject(err);
                })
                .run();
        } catch (err) {
            try { fs.unlinkSync(tempInput); } catch {}
            try { fs.unlinkSync(tempOutput); } catch {}
            reject(err);
        }
    });
}

async function extractAudioFeatures(audioBuffer: Buffer): Promise<{ pitch: number; loudness: number; snr: number }> {
    // Disabled (no AudioContext); returns zeros so callers keep working.
    console.warn("Audio feature extraction is disabled - AudioContext library not available");
    return { pitch: 0, loudness: 0, snr: 0 };
}

// Inputs for runGroupRound. nextNumber is the user turn's messageNumber; agent turns follow it.
interface PreparedGroupRound {
    agents: IAgent[];
    turnTaking: TurnTakingConfig;
    history: StoredMsg[];
    participantLabel: string;
    generate: (
        agent: IAgent,
        working: StoredMsg[],
        onChunk?: (chunk: string) => Promise<void>,
        extras?: { tailNote?: string },
    ) => Promise<string>;
    moderatorFn?: ModeratorFn;
    nextNumber: number;
    // Deterministic PRNG seed; undefined falls back to Math.random.
    seed?: string;
    // Phase per agent id, resolved once per round so an agent can't change character mid-round.
    phaseBySpeaker?: Record<string, PhaseResolution>;
    threadControl: ResolvedThreadControl;
    // First round of a freshly opened chat block.
    segmentJustOpened: boolean;
}

// A user round's state, read by withUserRound at the end to decide whether the user turn stays.
interface UserRound {
    // Null until persisted: nothing to take back.
    userTurn: { _id: unknown; messageNumber: number } | null;
    // Once messagesNumber is bumped the user turn stays, whatever fails afterwards.
    counted: boolean;
}

// 1-1 only: an empty completion (refusal, content filter, reasoning model out of budget) fails the
// round before anything is saved, so the user turn is taken back and Resend is offered.
// Group rounds handle empty generations in runGroupRound.
const assertReplyHasText = (reply: unknown): void => {
    if (typeof reply !== 'string' || !reply.trim()) {
        throw new AppError('The model returned an empty reply', 502, 'EMPTY_COMPLETION');
    }
};

// Maps a duplicate-key error on (conversation, messageNumber) to a 409: another writer (second
// instance, or a round that outlived the lock) used this number. The raw 11000 code must not
// reach a route as an HTTP status. Logged loudly since it's the only sign of a second writer.
const asMessageNumberConflict = (error: unknown, conversationId: string, messageNumber: number): unknown => {
    if ((error as { code?: unknown } | null)?.code !== 11000) return error;
    console.error(
        `[round] conversation ${conversationId}: message #${messageNumber} was refused — another writer ` +
            'has already used that number (a second server instance, or a round that outlived the ' +
            'conversation lock). The round fails, and the participant is offered Resend.',
    );
    return new AppError('This message could not be saved. Please send it again.', 409, 'MESSAGE_NUMBER_CONFLICT');
};

// Returns a copy with the client's message id; the caller's object goes to the provider and must not carry it.
const withClientMessageId = <M extends object>(message: M, clientMessageId?: string): M & { clientMessageId?: string } =>
    clientMessageId ? { ...message, clientMessageId } : message;

// Server-side budget for agent-only rounds, so a participant can't loop /round/continue. Refilled
// when the client refills its own: a counted user round, a block opening, an entry into an open block.
// Marker TTL equals the lock's max hold: past that the lock already let the next round through.
export const AUTONOMOUS_ROUND_MARKER_TTL_MS = CONVERSATION_LOCK_MAX_HOLD_MS;

// Read the same way the client does. Fractions are kept: R = 1.5 runs 2 rounds on both sides.
const autonomousBudgetOf = (turnTaking: TurnTakingConfig | undefined | null): number => {
    const rounds = Number(turnTaking?.autonomousRounds) || 0;
    return rounds > 0 ? rounds : 0;
};

// True while an agent-only round's in-flight marker is younger than the TTL.
const autonomousRoundMarkerIsFresh = (startedAt: unknown, now: number): boolean => {
    if (startedAt == null) return false;
    const at = startedAt instanceof Date ? startedAt.getTime() : new Date(startedAt as string | number).getTime();
    return Number.isFinite(at) && now - at < AUTONOMOUS_ROUND_MARKER_TTL_MS;
};

class ConversationsService {
    // Per-provider API key resolver for all generation paths. Overrides are read live, not
    // snapshotted, so a researcher can fix billing mid-study.
    private keyResolverFor = async (
        metadataConversation: any,
        callerRole: string,
        callerUserId: string,
    ): Promise<(provider: LLMProvider) => Promise<string>> => {
        const ownerId = apiKeysService.resolveKeyOwnerId(callerRole, callerUserId, metadataConversation.ownerId);
        const experiment = await experimentsService.getExperiment(metadataConversation.experimentId);
        const overrides = experiment?.apiKeyOverrides ?? {};
        return async (provider: LLMProvider) => {
            try {
                return await apiKeysService.resolveDecryptedKey(ownerId, provider, overrides[provider]);
            } catch (error) {
                throw this.audienceAwareKeyError(error, callerRole);
            }
        };
    };

    // Participants get a neutral message instead of "add a key in Settings". The reason stays
    // NO_API_KEY: the client treats it as terminal and releases a survey block cleanly.
    private audienceAwareKeyError = (error: any, callerRole: string) => {
        if (callerRole !== 'participant' || error?.reason !== 'NO_API_KEY') return error;
        return new AppError(
            'This conversation is unavailable right now. Please let the researcher know — there is nothing you need to do.',
            error.code ?? 403,
            'NO_API_KEY',
        );
    };

    // Phases: the agent snapshot already holds merged phases[]; these pick the current one.

    // Elapsed + turn count for the phase resolver. The anchor is the first round, not createdAt,
    // so time spent on the pre-conversation form doesn't skip phase 1.
    private resolvePhaseClock = async (
        conversationId: string,
        metadataConversation: any,
        userMessages: number,
    ): Promise<PhaseClock> => {
        const now = Date.now();
        const segmented = isSegmented(metadataConversation.segments);
        const current = segmented ? openSegment(metadataConversation.segments) : undefined;

        // Lazy stamp is unsegmented only; a segmented conversation anchors when its block opens.
        if (!segmented && !metadataConversation.phaseClockStartedAt) {
            // Mutate the in-memory copy too, or a later read in this round would re-stamp.
            metadataConversation.phaseClockStartedAt = now;
            // Write-once origin for the export's wall-clock column.
            metadataConversation.chatClockFirstStartedAt = now;
            await this.updateConversationMetadata(conversationId, {
                $set: { phaseClockStartedAt: now, chatClockFirstStartedAt: now },
            });
        }

        return {
            elapsedMs: chatElapsedMs(metadataConversation, now),
            userMessages,
            // Absent when unsegmented, so startAtSegment triggers stay inert.
            ...(current ? { segment: current.number } : {}),
        };
    };

    // Records the first time each agent enters a phase. Clock triggers can't be recomputed
    // from messageNumber, so the transition is stored for replay and analysis.
    private recordPhaseEntries = async (
        conversationId: string,
        metadataConversation: any,
        resolutions: { agentId: string; resolution: PhaseResolution }[],
        atMessageNumber: number,
        // Passed in, not recomputed, so the recorded elapsed equals the value the trigger fired on.
        clock?: PhaseClock,
    ): Promise<void> => {
        const history: any[] = metadataConversation.phaseHistory || [];
        const highest = new Map<string, number>();
        for (const h of history) highest.set(String(h.agentId), Math.max(highest.get(String(h.agentId)) ?? 0, h.phaseIndex));

        const entries = [];
        for (const { agentId, resolution } of resolutions) {
            if (resolution.index < 1) continue;
            const previous = highest.get(agentId) ?? 0;
            if (resolution.index <= previous) continue;
            entries.push({
                agentId,
                phaseIndex: resolution.index,
                phaseLabel: resolution.label ?? null,
                at: Date.now(),
                atMessageNumber,
                trigger: resolution.trigger,
                // Chat clock and segment kept separate from wall time; they differ in segmented studies.
                ...(clock ? { atChatElapsedMs: clock.elapsedMs } : {}),
                ...(clock?.segment != null ? { atSegment: clock.segment } : {}),
                // An idle participant can cross two boundaries between rounds; flag the skipped phase.
                ...(resolution.index > previous + 1 ? { skippedFrom: previous } : {}),
            });
        }
        if (!entries.length) return;

        for (const e of entries) {
            // +1: phaseIndex 1 is phase 2 (phase 1 is the base prompts), matching the export.
            console.log(
                `[phases] conversation ${conversationId}: agent ${e.agentId} entered phase ${e.phaseIndex + 1}` +
                    `${e.phaseLabel ? ` (${e.phaseLabel})` : ''} on ${e.trigger} at message ${atMessageNumber}` +
                    `${e.skippedFrom != null ? ` — SKIPPED phase(s) ${e.skippedFrom + 2}..${e.phaseIndex}` : ''}`,
            );
        }
        metadataConversation.phaseHistory = [...history, ...entries];
        await this.updateConversationMetadata(conversationId, { $push: { phaseHistory: { $each: entries } } });
    };

    // Humanized 1-1 agents are not token-streamed. Checked on a plain clone so a hydrated `{}`
    // pacing subdocument can't mis-trip the gate.
    isConversationHumanized = async (conversationId: string): Promise<boolean> => {
        const meta = await this.getConversationMetadata(conversationId);
        if (meta.mode === AgentsMode.GROUP) return false;
        return isHumanizedOneOnOne(JSON.parse(JSON.stringify(meta.agent)));
    };

    // Split replies can't be token-streamed: cut points need the whole reply. Must match the
    // `splitting` const in oneOnOneRound so controller and service agree on the response shape.
    isConversationSplit = async (conversationId: string): Promise<boolean> => {
        const meta = await this.getConversationMetadata(conversationId);
        if (meta.mode === AgentsMode.GROUP) return false;
        const agent = JSON.parse(JSON.stringify(meta.agent));
        return (
            isSplitEnabled(agent) &&
            agent.audioInput == null &&
            !(agent.cameraCaptureRate != null && agent.vaIntegration != null)
        );
    };

    // Used by the 1-1 SSE endpoint to reject group conversations.
    isConversationGroup = async (conversationId: string): Promise<boolean> => {
        const meta = await this.getConversationMetadata(conversationId);
        return meta.mode === AgentsMode.GROUP;
    };

    // streamResponse: the SSE writer, only when token-streaming. streamRequested: streaming was
    // asked for but this turn is buffered (a split reply).
    message = async (
        message,
        conversationId: string,
        callerUserId: string,
        callerRole: string,
        streamResponse?,
        streamRequested = false,
    ) => {
        // Kept off the turn, which goes to the provider as-is.
        const { clientMessageId, ...turn } = message;
        await this.assertNotAlreadySent(conversationId, clientMessageId);

        const [conversation, metadataConversation] = await Promise.all([
            this.getConversation(conversationId, true),
            this.getConversationMetadata(conversationId),
        ]);

        await this.assertCanSendTurn(conversationId, metadataConversation);

        if (metadataConversation.mode === AgentsMode.GROUP) {
            return this.groupMessage(turn, conversationId, callerUserId, callerRole, metadataConversation, clientMessageId);
        }

        // withUserRound takes the user turn back out if the round fails before the reply is saved.
        return this.withUserRound(conversationId, (round) =>
            this.oneOnOneRound(
                round,
                turn,
                conversationId,
                callerUserId,
                callerRole,
                conversation,
                metadataConversation,
                streamResponse,
                streamRequested,
                clientMessageId,
            ),
        );
    };

    // Refuses a re-send of a stored message (client re-sync after a dropped stream). Checked first,
    // so a stored message is never reported as refused by the cap or a closed block.
    private assertNotAlreadySent = async (conversationId: string, clientMessageId?: string): Promise<void> => {
        if (!clientMessageId) return;
        if (await ConversationsModel.exists({ conversationId, role: 'user', clientMessageId })) {
            throw new AppError('This message has already been sent.', 409, 'DUPLICATE_MESSAGE');
        }
    };

    // The single-agent / A-B round behind message(), run inside withUserRound.
    private oneOnOneRound = async (
        round: UserRound,
        message,
        conversationId: string,
        callerUserId: string,
        callerRole: string,
        conversation: Message[],
        metadataConversation: any,
        streamResponse?,
        streamRequested = false,
        // Written on the user turn only, never sent to the provider.
        clientMessageId?: string,
    ) => {
        const snapshotAgent = JSON.parse(JSON.stringify(metadataConversation.agent));
        // Phases only swap prompts, so resolving before the pacing gates changes nothing else.
        // +1: the user turn being answered isn't persisted yet.
        let onePhase: PhaseResolution | undefined;
        if (hasPhases([snapshotAgent])) {
            const userMessages = conversation.filter((m: any) => m.role === 'user').length + 1;
            const clock = await this.resolvePhaseClock(conversationId, metadataConversation, userMessages);
            onePhase = resolvePhase(snapshotAgent.phases, clock);
            await this.recordPhaseEntries(
                conversationId,
                metadataConversation,
                [{ agentId: String(snapshotAgent._id), resolution: onePhase }],
                conversation.length + 1,
                clock,
            );
        }
        // Knowledge goes on after the phase: a phase replaces the system prompt wholesale.
        const agent = applyKnowledge(
            onePhase ? applyPhase(snapshotAgent, onePhase.index) : snapshotAgent,
            composeKnowledge(metadataConversation.knowledge?.[String(snapshotAgent._id)]),
        );
        // Stamped for any phased conversation, index 0 included, so absence means "no phases".
        const phaseStamp = onePhase ? { index: onePhase.index, label: onePhase.label ?? null } : undefined;
        //const { cameraCaptureRate, ...agentWithoutCameraCaptureRate } = agent;
        //console.log(agent)
        const ccr = agent.cameraCaptureRate
        const vai = agent.vaIntegration
        // Effective typing delay: it lives outside the pacing object, so its toggle is applied here.
        const timeDelay = resolveTypingDelay(agent)
        const prev_message = message.content;
        // Same gate as isConversationHumanized. ccr/vai are deleted from agent below, so pass them.
        const humanized = isHumanizedOneOnOne({ ...agent, cameraCaptureRate: ccr, vaIntegration: vai });
        // Splitting is independent of HD but shares its exclusions: audio is one utterance, and the
        // explainable turn's second completion at len+2 would collide with split documents.
        const splitting = isSplitEnabled(agent) && agent.audioInput == null && !(ccr != null && vai != null);
        // Non-HD 1-1 replies get a short beat and typing indicator. Not when streaming: the text is
        // already appearing. Split replies return earlier and pace themselves.
        const paceDefault =
            !humanized &&
            !streamResponse &&
            !streamRequested &&
            agent.audioInput == null &&
            !(ccr != null && vai != null);

        //console.log("vai", vai)
        //console.log("ccr", ccr)
        delete agent.cameraCaptureRate;
        delete agent.vaIntegration;
        //let tempt = await CurrentStateModels.find({ }).exec();
        //console.log(tempt)
        let val = 0;
        let ar = 0

        if ( ccr != null && vai != null ) {
            const current_state = await this.getCurrentState(conversationId)
            val = current_state[0]["valence"] / current_state[0]["count"]
            ar = current_state[0]["arousal"] / current_state[0]["count"]
        }
        //console.log(current_state[0]["valence"])
        //console.log(current_state[0]["arousal"])

        const og_text = { ...message };
        //console.log(og_text)
        const messages: any[] = this.getConversationMessages(agent, conversation, message, val, ar, ccr, vai);
        const chatRequest = this.getChatRequest(agent, messages);
        const userTurn = await this.createMessageDoc(
            withClientMessageId(message, clientMessageId),
            conversationId,
            conversation.length + 1,
            val, ar, 0, 0, 0,
        );
        // Until countUserRound, this turn is taken back if the round fails.
        round.userTurn = { _id: userTurn._id, messageNumber: conversation.length + 1 };

        const provider = getProviderFromModel(agent.model);
        const resolveKey = await this.keyResolverFor(metadataConversation, callerRole, callerUserId);
        const decryptedKey = await resolveKey(provider);

        // HD and split replies are generated buffered; must agree with the controller's gate.
        let assistantMessage = '';
        if (streamResponse && !humanized && !splitting) {
            assistantMessage = await createChatCompletion(decryptedKey, provider, chatRequest, true, streamResponse);
        } else {
            assistantMessage = await createChatCompletion(decryptedKey, provider, chatRequest, false);
        }
        // Before either persist branch. A streamed whitespace reply is replaced by the error frame.
        assertReplyHasText(assistantMessage);

        if (humanized || splitting) {
            // Persist each paced segment as its own doc (len+2, +3, ...); messagesNumber is bumped
            // once. Returns the segment array, possibly of one.
            const baseNumber = conversation.length + 2;
            const seed = metadataConversation.seed;
            const splitRng =
                seed != null ? makeRng(seed, `split:${baseNumber}`) : mulberry32(Math.floor(Math.random() * 2 ** 31));
            const segs = buildPacedSegments({
                reply: assistantMessage,
                prevContent: prev_message,
                pacing: agent.pacing,
                timeDelay: timeDelay ?? null,
                splitRng,
                // Timing stream keyed per segment.
                timeRngFor: (i) =>
                    seed != null
                        ? makeRng(seed, `time:${baseNumber}:${i}`)
                        : mulberry32(Math.floor(Math.random() * 2 ** 31)),
                segmentGroupId: `seg-s-${baseNumber}`,
                // Pass the real verdict: omitted it defaults to true and a split-only agent
                // would get the full HD timing envelope.
                humanized,
            });
            // If streaming was requested the client typewrites each bubble: drop the typing delays,
            // keep the gap between segments.
            const savedSegments: Message[] = [];
            for (let i = 0; i < segs.length; i++) {
                const seg = segs[i];
                savedSegments.push(
                    await this.createMessageDoc(
                        { role: 'assistant', content: seg.content, timeDelay: seg.timeDelay, prev_message: seg.prev_message },
                        conversationId,
                        baseNumber + i,
                        val,
                        ar,
                        0,
                        0,
                        0,
                        {
                            preTypingDelayMs: streamRequested ? 0 : seg.preTypingDelayMs,
                            typingDurationMs: streamRequested ? 0 : seg.typingDurationMs,
                            interSegmentPauseMs: seg.interSegmentPauseMs,
                            segmentGroupId: seg.segmentGroupId,
                            segmentIndex: seg.segmentIndex,
                            segmentCount: seg.segmentCount,
                        },
                        phaseStamp,
                    ),
                );
            }
            await this.countUserRound(conversationId, round);
            await this.evaluateSegmentAfterRound(conversationId, metadataConversation);
            return savedSegments;
        }

        // One unsplit segment with the default-cadence envelope. Throwaway rngs: nothing here
        // draws from the conversation's seeded streams.
        const defaultEnv = paceDefault
            ? buildPacedSegments({
                  reply: assistantMessage,
                  prevContent: prev_message,
                  pacing: NO_OP_PACING,
                  timeDelay: timeDelay ?? null,
                  splitRng: mulberry32(1),
                  timeRngFor: () => mulberry32(1),
                  segmentGroupId: `seg-s-${conversation.length + 2}`,
                  humanized: false,
              })[0]
            : null;

        const savedMessage = await this.createMessageDoc(
            {
                content: assistantMessage,
                role: 'assistant',
                timeDelay: timeDelay,
                prev_message: prev_message,
            },
            conversationId,
            conversation.length + 2,
            val,
            ar,
            0,
            0,
            0,
            defaultEnv
                ? {
                      preTypingDelayMs: defaultEnv.preTypingDelayMs,
                      typingDurationMs: defaultEnv.typingDurationMs,
                      interSegmentPauseMs: defaultEnv.interSegmentPauseMs,
                      // Not a split reply.
                      segmentGroupId: null,
                      segmentIndex: null,
                      segmentCount: null,
                  }
                : undefined,
            phaseStamp,
        );

        await this.countUserRound(conversationId, round);
        await this.evaluateSegmentAfterRound(conversationId, metadataConversation);

        if ( ccr != null && vai != null ) {
            const Exmessages: any[] = this.getExplainableText(agent, conversation, message, val, ar);
            const ExchatRequest = this.getChatRequest(agent, Exmessages);

            const ExassistantMessage = await createChatCompletion(decryptedKey, provider, ExchatRequest, false);

            console.log("idk why", og_text)
            await this.createExplainableDoc(
                og_text,
                message,
                {
                    content: ExassistantMessage,
                    role: 'assistant',
                    timeDelay: timeDelay
                },
                conversationId,
                conversation.length + 2,
                val,
                ar,
            );
        }

        return savedMessage;
    };

    // Group round, batch: returns all agent replies at once. No audio / VA / explainable.
    private groupMessage = async (
        message,
        conversationId: string,
        callerUserId: string,
        callerRole: string,
        metadataConversation: any,
        clientMessageId?: string,
    ): Promise<Message[]> =>
        // Setup writes the user turn, so it runs inside withUserRound too.
        this.withUserRound(conversationId, async (round) => {
            const { agents, turnTaking, history, participantLabel, generate, moderatorFn, nextNumber, seed, phaseBySpeaker, threadControl, segmentJustOpened } =
                await this.prepareGroupRound(message.content, conversationId, callerUserId, callerRole, metadataConversation, round, clientMessageId);

            // Same `u<n>` anchor as the streaming path, so this fallback reproduces its draws.
            const roundAnchor = `u${nextNumber}`;
            const newMessages = await runGroupRound({
                agents,
                history,
                turnTaking,
                participantLabel,
                generate,
                moderatorFn,
                seed,
                roundAnchor,
                threadControl,
                segmentJustOpened,
                onRoundEvent: (e) =>
                    this.recordThreadClose(conversationId, metadataConversation, nextNumber, roundAnchor, e),
                // A silent round still completes: the user turn stands and counts.
                onSilentRound: (e) =>
                    this.recordSilentRound(conversationId, metadataConversation, nextNumber, roundAnchor, e.agentIds),
            });

            let number = nextNumber;
            const saved: Message[] = [];
            for (const m of newMessages) {
                number += 1;
                saved.push(await this.persistGroupTurn(m, conversationId, number, phaseBySpeaker));
            }

            // One count per user round, zero agent turns included.
            await this.countUserRound(conversationId, round);
            await this.evaluateSegmentAfterRound(conversationId, metadataConversation);

            return saved;
        });

    // Group token streaming only when no agent is humanized (HD wins). Splitters still stream;
    // they buffer their own turn. `agents` is already a plain clone.
    private groupStreamAllowed = (agents: IAgent[], requested: boolean): boolean =>
        requested && agents.length > 0 && !agents.some((a) => isHumanized(a));

    // Group round, streaming: persists and emits each turn as produced.
    // onAccepted opens the NDJSON stream after all refusals, so a refusal is a plain JSON error
    // with its real status, not a 200 with an error line.
    groupMessageStream = async (
        message,
        conversationId: string,
        callerUserId: string,
        callerRole: string,
        onEmit: (saved: Message) => Promise<void>,
        stream = false,
        onDelta?: (
            meta: { turnId: string; speakerId: string; speakerName: string; interSegmentPauseMs?: number },
            chunk: string,
        ) => Promise<void>,
        onAccepted?: () => void,
    ): Promise<void> => {
        // Refusals in order: duplicate, not group, closed block / message cap. All under the lock.
        await this.assertNotAlreadySent(conversationId, message.clientMessageId);

        const metadataConversation = await this.getConversationMetadata(conversationId);

        if (metadataConversation.mode !== AgentsMode.GROUP) {
            throw new AppError('Not a group conversation', 400, 'NOT_GROUP_CONVERSATION');
        }
        await this.assertCanSendTurn(conversationId, metadataConversation);

        // From here failures arrive as the stream's NDJSON error line.
        onAccepted?.();

        // A streamed round can fail with a reply already saved; withUserRound then keeps the user turn.
        await this.withUserRound(conversationId, async (round) => {
            const { agents, turnTaking, history, participantLabel, generate, moderatorFn, nextNumber, seed, phaseBySpeaker, threadControl, segmentJustOpened } =
                await this.prepareGroupRound(message.content, conversationId, callerUserId, callerRole, metadataConversation, round, message.clientMessageId);

            let number = nextNumber;
            const roundAnchor = `u${nextNumber}`;
            await runGroupRound({
                agents,
                history,
                turnTaking,
                participantLabel,
                generate,
                moderatorFn,
                seed,
                roundAnchor,
                threadControl,
                segmentJustOpened,
                onRoundEvent: (e) =>
                    this.recordThreadClose(conversationId, metadataConversation, nextNumber, roundAnchor, e),
                onSilentRound: (e) =>
                    this.recordSilentRound(conversationId, metadataConversation, nextNumber, roundAnchor, e.agentIds),
                onTurnChunk: this.groupStreamAllowed(agents, stream) ? onDelta : undefined,
                onTurn: async (m) => {
                    number += 1;
                    const saved = await this.persistGroupTurn(m, conversationId, number, phaseBySpeaker);
                    await onEmit(saved);
                },
            });

            await this.countUserRound(conversationId, round);
            await this.evaluateSegmentAfterRound(conversationId, metadataConversation);
        });
    };

    // One agent-only round, streamed. Writes no user turn and doesn't count against the user cap.
    // The client decides when; the server budget decides whether. onAccepted as in groupMessageStream.
    autonomousGroupRoundStream = async (
        conversationId: string,
        callerUserId: string,
        callerRole: string,
        metadataConversation: any,
        onEmit: (saved: Message) => Promise<void>,
        stream = false,
        onDelta?: (
            meta: { turnId: string; speakerId: string; speakerName: string; interSegmentPauseMs?: number },
            chunk: string,
        ) => Promise<void>,
        onAccepted?: () => void,
    ): Promise<void> => {
        if (metadataConversation.mode !== AgentsMode.GROUP) {
            throw new AppError('Not a group conversation', 400, 'NOT_GROUP_CONVERSATION');
        }
        const budget = this.assertCanRunAutonomousRound(metadataConversation);
        const claimedAt = await this.claimAutonomousRound(conversationId, budget);

        try {
            onAccepted?.();

            // No user turn and no UserRound: the previous user turn must never be taken back.
            const { agents, turnTaking, history, participantLabel, generate, moderatorFn, nextNumber, seed, phaseBySpeaker, threadControl, segmentJustOpened } =
                await this.prepareGroupRound(null, conversationId, callerUserId, callerRole, metadataConversation, null);

            let number = nextNumber;
            // `a` prefix: autonomous and user rounds derive nextNumber differently and could collide.
            const roundAnchor = `a${nextNumber}`;
            await runGroupRound({
                agents,
                history,
                turnTaking,
                participantLabel,
                generate,
                moderatorFn,
                seed,
                roundAnchor,
                threadControl,
                segmentJustOpened,
                onRoundEvent: (e) =>
                    this.recordThreadClose(conversationId, metadataConversation, nextNumber, roundAnchor, e),
                // Every turn sees the previous ones; no blind first wave.
                roundKind: 'autonomous',
                onTurnChunk: this.groupStreamAllowed(agents, stream) ? onDelta : undefined,
                onTurn: async (m) => {
                    number += 1;
                    const saved = await this.persistGroupTurn(m, conversationId, number, phaseBySpeaker);
                    await onEmit(saved);
                },
            });

            // Timestamps only; messagesNumber is not bumped.
            await this.updateConversationMetadata(conversationId, {
                $set: { lastMessageDate: new Date(), lastMessageTimestamp: Date.now() },
            });
        } finally {
            // The claimed slot stays spent even on failure, as on the client.
            await this.releaseAutonomousRound(conversationId, claimedAt);
        }

        // A time ceiling can fall due during an agent-only round. Evaluated after the release so
        // a survey entry arriving now finds the round over.
        await this.evaluateSegmentAfterRound(conversationId, metadataConversation);
    };

    // Refusals for an agent-only round, most final first; returns the budget.
    // Not checked: maxMessages (agent-only rounds don't count), and autonomousLevel, which is a
    // per-round turn cap rather than an on/off switch.
    private assertCanRunAutonomousRound = (metadataConversation: any): number => {
        if (metadataConversation.isFinished) {
            throw new AppError('This conversation has ended.', 403, 'CONVERSATION_FINISHED');
        }
        // Same rule as assertCanSendTurn.
        if (isSegmented(metadataConversation.segments) && !openSegment(metadataConversation.segments)) {
            throw new AppError('This part of the conversation has ended.', 403, 'SEGMENT_CLOSED');
        }
        const budget = autonomousBudgetOf(metadataConversation.turnTaking);
        if (budget <= 0) {
            throw new AppError('Agent-only rounds are not enabled for this conversation.', 403, 'AUTONOMOUS_DISABLED');
        }
        return budget;
    };

    // Atomically checks and spends the budget and raises the in-flight marker; returns its stamp.
    // Atomic so it holds across server instances, not just under the in-process lock.
    // No match means budget spent or another round in flight: 429.
    private claimAutonomousRound = async (conversationId: string, budget: number, now = Date.now()): Promise<Date> => {
        const startedAt = new Date(now);
        const claimed = await MetadataConversationsModel.findOneAndUpdate(
            {
                _id: new mongoose.Types.ObjectId(conversationId),
                $and: [
                    { $or: [{ autonomousRoundsUsed: { $exists: false } }, { autonomousRoundsUsed: { $lt: budget } }] },
                    {
                        $or: [
                            { autonomousRoundStartedAt: { $exists: false } },
                            { autonomousRoundStartedAt: { $lt: new Date(now - AUTONOMOUS_ROUND_MARKER_TTL_MS) } },
                        ],
                    },
                ],
            },
            { $inc: { autonomousRoundsUsed: 1 }, $set: { autonomousRoundStartedAt: startedAt } },
            { new: true, projection: { _id: 1 } },
        ).lean();
        if (!claimed) {
            throw new AppError(
                'No more agent-only rounds until the participant speaks again.',
                429,
                'AUTONOMOUS_BUDGET_EXHAUSTED',
            );
        }
        return startedAt;
    };

    // Clears only our own marker (matched by stamp), so an overrun round can't clear a later one's.
    // Never throws: runs in a finally and must not replace the round's error. A stuck marker expires.
    private releaseAutonomousRound = async (conversationId: string, startedAt: Date): Promise<void> => {
        try {
            await MetadataConversationsModel.updateOne(
                { _id: new mongoose.Types.ObjectId(conversationId), autonomousRoundStartedAt: startedAt },
                { $unset: { autonomousRoundStartedAt: '' } },
            );
        } catch (error) {
            console.error(
                `[round] conversation ${conversationId}: could not clear the agent-only round's in-flight ` +
                    `marker — survey entries will defer for up to ${Math.round(AUTONOMOUS_ROUND_MARKER_TTL_MS / 1000)}s: ${error}`,
            );
        }
    };

    // Refill the agent-only budget. Writes nothing if no agent-only round ever ran.
    private refillAutonomousBudget = async (conversationId: string): Promise<void> => {
        await MetadataConversationsModel.updateOne(
            { _id: new mongoose.Types.ObjectId(conversationId), autonomousRoundsUsed: { $exists: true } },
            { $unset: { autonomousRoundsUsed: '' } },
        );
    };

    // Shared group-round setup: persists the user turn, builds history, keys, generate() and moderator.
    // `round` is null for an autonomous round. The user turn is registered on it as soon as it's
    // written, since later setup (key resolution) can still fail.
    private prepareGroupRound = async (
        userMessage: string | null,
        conversationId: string,
        callerUserId: string,
        callerRole: string,
        metadataConversation: any,
        round: UserRound | null,
        clientMessageId?: string,
    ): Promise<PreparedGroupRound> => {
        // Base prompts; `agents` gets the active phase applied.
        const snapshotAgents: IAgent[] = JSON.parse(JSON.stringify(metadataConversation.agents || []));
        let agents: IAgent[] = snapshotAgents;
        let phaseBySpeaker: Record<string, PhaseResolution> | undefined;
        const turnTaking: TurnTakingConfig = metadataConversation.turnTaking || { policy: 'rules' };
        // Older conversations have no seed; backfill it so later rounds are reproducible.
        // seedBackfilledAtMessage marks where reproducibility starts.
        let seed: string | undefined = metadataConversation.seed;
        if (!seed) {
            seed = crypto.randomBytes(8).toString('hex');
            await this.updateConversationMetadata(conversationId, {
                $set: { seed, seedBackfilledAtMessage: metadataConversation.messagesNumber ?? 0 },
            });
        }

        const existingDocs = await ConversationsModel.find(
            { conversationId },
            {
                _id: 0,
                role: 1,
                content: 1,
                speakerId: 1,
                speakerName: 1,
                timeDelay: 1,
                prev_message: 1,
                messageNumber: 1,
                // So an earlier split turn reads as one turn for mention detection.
                segmentGroupId: 1,
            },
        )
            .sort({ messageNumber: 1 })
            .lean();

        // nextNumber is the number before the first agent turn. Autonomous rounds write no user turn.
        let history: StoredMsg[];
        let nextNumber: number;
        if (userMessage != null) {
            nextNumber = existingDocs.length + 1;
            const userTurn = await this.createMessageDoc(
                withClientMessageId({ role: 'user', content: userMessage, timeDelay: null, prev_message: null }, clientMessageId),
                conversationId,
                nextNumber,
                0, 0, 0, 0, 0,
            );
            if (round) round.userTurn = { _id: userTurn._id, messageNumber: nextNumber };
            history = assembleHistory([...existingDocs, { role: 'user', content: userMessage }]);
        } else {
            nextNumber = existingDocs.length;
            history = assembleHistory([...existingDocs]);
        }

        // Count user turns from the transcript: messagesNumber isn't bumped until the round ends.
        // Autonomous rounds advance the clock but not the count.
        if (hasPhases(snapshotAgents)) {
            const userMessages =
                existingDocs.filter((d: any) => d.role === 'user').length + (userMessage != null ? 1 : 0);
            const clock = await this.resolvePhaseClock(conversationId, metadataConversation, userMessages);
            const resolutions = snapshotAgents.map((a) => ({
                agentId: String(a._id),
                resolution: resolvePhase(a.phases, clock),
            }));
            phaseBySpeaker = Object.fromEntries(resolutions.map((r) => [r.agentId, r.resolution]));
            // The only place phases enter the group path: everything downstream reads prompts off the agent.
            agents = snapshotAgents.map((a) => applyPhase(a, phaseBySpeaker![String(a._id)].index));
            await this.recordPhaseEntries(conversationId, metadataConversation, resolutions, nextNumber, clock);
        }

        // Per-agent survey knowledge, applied after the phase (which replaces the system prompt).
        if (metadataConversation.knowledge) {
            agents = agents.map((a) =>
                applyKnowledge(a, composeKnowledge(metadataConversation.knowledge?.[String(a._id)])),
            );
        }

        const resolveKey = await this.keyResolverFor(metadataConversation, callerRole, callerUserId);
        const providerOf = (agent: IAgent) => getProviderFromModel(agent.model);
        const providers = [...new Set(agents.map(providerOf))];
        const keyEntries = await Promise.all(
            providers.map(async (p) => [p, await resolveKey(p)] as const),
        );
        const keyMap: Record<string, string> = Object.fromEntries(keyEntries);

        const user = await usersService.getUserById(metadataConversation.userId);
        const participantLabel = user?.username || 'User';

        // onChunk turns on provider streaming. Scene and window come from the snapshot so
        // mid-study edits don't reach running conversations.
        const scene = resolveScene(turnTaking);
        const memoryWindow = resolveMemoryWindow(turnTaking);
        const generate = (
            agent: IAgent,
            working: StoredMsg[],
            onChunk?: (chunk: string) => Promise<void>,
            extras?: { tailNote?: string },
        ) =>
            createGroupChatCompletion(
                keyMap[providerOf(agent)],
                providerOf(agent),
                agent,
                working,
                agents,
                participantLabel,
                turnTaking.identifyHumanParticipant ?? false,
                !!onChunk,
                onChunk,
                scene,
                extras?.tailNote,
                memoryWindow,
            );

        // 'llm' policy moderator uses the first agent's model; the group completion gives Gemini
        // thinking models their output floor.
        let moderatorFn: ModeratorFn | undefined;
        if (turnTaking.policy === 'llm' && agents.length) {
            const moderatorAgent = agents[0];
            const moderatorProvider = providerOf(moderatorAgent);
            const moderatorKey = keyMap[moderatorProvider];
            moderatorFn = (system: string, transcript: string) =>
                createModeratorCompletion(moderatorKey, moderatorProvider, moderatorAgent.model, system, transcript);
        }

        // An autonomous round in a block where nothing has been said yet, so the agents open it.
        // Blocks without openedAtTranscriptPosition keep the legacy comparison.
        const openBlock = openSegment(metadataConversation.segments);
        const segmentJustOpened =
            userMessage == null &&
            isSegmented(metadataConversation.segments) &&
            !!openBlock &&
            (typeof openBlock.openedAtTranscriptPosition === 'number'
                ? openBlock.openedAtTranscriptPosition === transcriptPositionOf(existingDocs)
                : openBlock.openedAtMessageNumber === existingDocs.length);

        return {
            agents,
            turnTaking,
            history,
            participantLabel,
            generate,
            moderatorFn,
            nextNumber,
            seed,
            phaseBySpeaker,
            threadControl: resolveThreadControl(turnTaking),
            segmentJustOpened,
        };
    };

    // Records a round thread control ended early; the transcript alone can't show it.
    private recordThreadClose = async (
        conversationId: string,
        metadataConversation: any,
        atMessageNumber: number,
        roundAnchor: string,
        event: { by: 'draw' | 'guard'; turnIndex: number },
    ): Promise<void> => {
        const entry = {
            atMessageNumber,
            roundAnchor,
            turnIndex: event.turnIndex,
            by: event.by,
            at: Date.now(),
        };
        console.log(
            `[thread] conversation ${conversationId}: round ${roundAnchor} closed at turn ${event.turnIndex} ` +
                `by ${event.by === 'guard' ? 'the degeneration guard' : 'the seeded draw'}`,
        );
        metadataConversation.threadLog = [...(metadataConversation.threadLog || []), entry];
        await this.updateConversationMetadata(conversationId, { $push: { threadLog: entry } });
    };

    // Records a user round with no reply because every agent returned empty, so the export can
    // tell a provider failure from designed silence (moderator NONE, silenceChance), which isn't recorded.
    private recordSilentRound = async (
        conversationId: string,
        metadataConversation: any,
        messageNumber: number,
        roundAnchor: string,
        agentIds: string[],
    ): Promise<void> => {
        const entry: SilentRoundRecord = { messageNumber, at: Date.now(), agentIds };
        console.warn(
            `[round] conversation ${conversationId}: participant message #${messageNumber} (round ${roundAnchor}) ` +
                `got no reply — every agent asked returned an empty one (${agentIds.join(', ')}). Recorded as a silent round.`,
        );
        metadataConversation.silentRounds = [...(metadataConversation.silentRounds || []), entry];
        await this.updateConversationMetadata(conversationId, { $push: { silentRounds: entry } });
    };

    // Persists one agent turn and returns it with speakerId/speakerName.
    private persistGroupTurn = async (
        turn: StoredMsg,
        conversationId: string,
        number: number,
        // Resolved at round start, so all segments of a split turn share one phase.
        phaseBySpeaker?: Record<string, PhaseResolution>,
    ): Promise<Message> => {
        const phase = phaseBySpeaker?.[String(turn.speakerId)];
        const savedMsg = await this.createMessageDoc(
            {
                role: 'assistant',
                content: turn.content,
                speakerId: turn.speakerId,
                speakerName: turn.speakerName,
                timeDelay: turn.timeDelay,
                prev_message: turn.prev_message,
            },
            conversationId,
            number,
            0, 0, 0, 0, 0,
            // Timing envelope from runGroupRound; must be passed via `extra` to reach the client.
            {
                preTypingDelayMs: turn.preTypingDelayMs,
                typingDurationMs: turn.typingDurationMs,
                interSegmentPauseMs: turn.interSegmentPauseMs,
                segmentGroupId: turn.segmentGroupId,
                segmentIndex: turn.segmentIndex,
                segmentCount: turn.segmentCount,
                revealAtMs: turn.revealAtMs,
            },
            phase ? { index: phase.index, label: phase.label ?? null } : undefined,
            // Stamped on every segment of a split turn.
            turn.move ?? undefined,
        );
        return { ...savedMsg, speakerId: turn.speakerId, speakerName: turn.speakerName };
    };

    audio = async (message, conversationId: string, callerUserId: string, callerRole: string, streamResponse?) => {
        // Duplicate check before transcription, which is a billed Whisper call.
        const { clientMessageId, ...turn } = message;
        await this.assertNotAlreadySent(conversationId, clientMessageId);

        const [conversation, metadataConversation] = await Promise.all([
            this.getConversation(conversationId, true),
            this.getConversationMetadata(conversationId),
        ]);

        await this.assertCanSendTurn(conversationId, metadataConversation);

        return this.withUserRound(conversationId, (round) =>
            this.audioRound(
                round,
                turn,
                conversationId,
                callerUserId,
                callerRole,
                conversation,
                metadataConversation,
                streamResponse,
                clientMessageId,
            ),
        );
    };

    // The round behind audio(), run inside withUserRound.
    private audioRound = async (
        round: UserRound,
        message,
        conversationId: string,
        callerUserId: string,
        callerRole: string,
        conversation: Message[],
        metadataConversation: any,
        streamResponse?,
        // Written on the user turn only.
        clientMessageId?: string,
    ) => {
        const snapshotAgent = JSON.parse(JSON.stringify(metadataConversation.agent));
        // Phases, as in oneOnOneRound. +1: the user turn isn't persisted yet.
        let onePhase: PhaseResolution | undefined;
        if (hasPhases([snapshotAgent])) {
            const userMessages = conversation.filter((m: any) => m.role === 'user').length + 1;
            const clock = await this.resolvePhaseClock(conversationId, metadataConversation, userMessages);
            onePhase = resolvePhase(snapshotAgent.phases, clock);
            await this.recordPhaseEntries(
                conversationId,
                metadataConversation,
                [{ agentId: String(snapshotAgent._id), resolution: onePhase }],
                conversation.length + 1,
                clock,
            );
        }
        // Knowledge goes on after the phase: a phase replaces the system prompt wholesale.
        const agent = applyKnowledge(
            onePhase ? applyPhase(snapshotAgent, onePhase.index) : snapshotAgent,
            composeKnowledge(metadataConversation.knowledge?.[String(snapshotAgent._id)]),
        );
        const phaseStamp = onePhase ? { index: onePhase.index, label: onePhase.label ?? null } : undefined;
        const ccr = agent.cameraCaptureRate
        const vai = agent.vaIntegration
        const timeDelay = agent.inverseTimeDelay

        delete agent.cameraCaptureRate;
        delete agent.vaIntegration;

        //console.log(message.content)
        //console.log(typeof(message.content))

        //const audioBlob = new Blob([message.content], { type: "audio/wav" });

        let pit = 0
        let loud = 0
        let sn = 0

        await this.processAudio(message.content).then(({ pitch, loudness, snr }) => {
            pit = pitch
            loud = loudness
            sn = snr
        });

        console.log(pit)
        console.log(loud)
        console.log(sn)

        // Whisper/TTS always need an OpenAI key, whatever the chat provider.
        const provider = getProviderFromModel(agent.model);
        const resolveKey = await this.keyResolverFor(metadataConversation, callerRole, callerUserId);
        const openaiKey = await resolveKey('openai');
        const chatKey = provider === 'openai' ? openaiKey : await resolveKey(provider);

        const text = await this.transcribeAudio(message.content, openaiKey);

        message.content = text;
        const prev_message = message.content;

        let val = 0;
        let ar = 0

        if ( ccr != null && vai != null ) {
            const current_state = await this.getCurrentState(conversationId)
            val = current_state[0]["valence"] / current_state[0]["count"]
            ar = current_state[0]["arousal"] / current_state[0]["count"]
        }

        const og_text = { ...message };
        const messages: any[] = this.getConversationMessages(agent, conversation, message, val, ar, ccr, vai);
        const chatRequest = this.getChatRequest(agent, messages);

        const userTurn = await this.createMessageDoc(
            withClientMessageId(message, clientMessageId),
            conversationId,
            conversation.length + 1,
            val, ar, pit, loud, sn,
        );
        round.userTurn = { _id: userTurn._id, messageNumber: conversation.length + 1 };

        let assistantMessage = '';
        if (!streamResponse) {
            assistantMessage = await createChatCompletion(chatKey, provider, chatRequest, false);
        } else {
            assistantMessage = await createChatCompletion(chatKey, provider, chatRequest, true, streamResponse);
        }
        // Also keeps an empty reply away from TTS.
        assertReplyHasText(assistantMessage);

        const savedMessage = await this.createMessageDoc(
            {
                content: assistantMessage,
                role: 'assistant',
                timeDelay: timeDelay,
                prev_message: prev_message,
            },
            conversationId,
            conversation.length + 2,
            val,
            ar,
            pit,
            loud,
            sn,
            undefined,
            phaseStamp,
        );

        // Counted before TTS: the reply is saved, so a TTS failure keeps the user turn.
        await this.countUserRound(conversationId, round);
        await this.evaluateSegmentAfterRound(conversationId, metadataConversation);

        const audio_new = await this.transcribeText(assistantMessage, openaiKey);
        console.log(audio_new)

        if ( ccr != null && vai != null ) {
            const Exmessages: any[] = this.getExplainableText(agent, conversation, message, val, ar);
            const ExchatRequest = this.getChatRequest(agent, Exmessages);

            const ExassistantMessage = await createChatCompletion(chatKey, provider, ExchatRequest, false);

            await this.createExplainableDoc(
                og_text,
                message,
                {
                    content: ExassistantMessage,
                    role: 'assistant',
                    timeDelay: timeDelay
                },
                conversationId,
                conversation.length + 2,
                val,
                ar,
            );
        }
        
        const newMessage = {
            "_id": savedMessage._id, 
            "role": savedMessage.role,
            "content": audio_new, 
            "userAnnotation": savedMessage.userAnnotation,
            "timeDelay": savedMessage.timeDelay,
        }
        //console.log(newMessage)
        return newMessage;
    };

    // Segment persistence. The decisions live in segments.service.

    // Record for a block opening now. The quota is frozen here so mid-study edits don't
    // affect an open block.
    private buildSegmentRecord = (
        number: number,
        experimentSegments: SegmentsConfig | undefined,
        now: number,
        messagesNumber: number,
        userMessages: number,
        transcriptPosition: number,
        skippedFrom?: number,
    ): SegmentRecord => ({
        number,
        openedAt: now,
        // Explicit null: same to Mongo as missing, easier to see in a dump.
        closedAt: null,
        // Legacy round count, kept for old readers and exports.
        openedAtMessageNumber: messagesNumber,
        openedAtTranscriptPosition: transcriptPosition,
        openedAtUserMessages: userMessages,
        quota: quotaForSegment(experimentSegments, number),
        ...(skippedFrom != null ? { skippedFrom } : {}),
    });

    // Highest persisted messageNumber (0 if none), recorded when a block opens or closes.
    // Unlocked: segment routes skip the round lock, but every open happens when no round is
    // adding to the transcript (creation, a round boundary, or no open block).
    private readTranscriptPosition = async (conversationId: string): Promise<number> => {
        const last: any = await ConversationsModel.findOne({ conversationId }, { _id: 0, messageNumber: 1 })
            .sort({ messageNumber: -1 })
            .lean();
        return typeof last?.messageNumber === 'number' ? last.messageNumber : 0;
    };

    // Close the open block and bank its span into the chat clock. Idempotent: the filter needs
    // a still-open element, so a racing second close can't double-bank the clock.
    closeOpenSegment = async (
        conversationId: string,
        metadataConversation: any,
        reason: SegmentCloseReason,
        now = Date.now(),
    ): Promise<SegmentRecord | null> => {
        const current = openSegment(metadataConversation?.segments);
        if (!current) return null;

        // Only if the open recorded one too, so both ends use the same unit.
        const closedAtTranscriptPosition =
            typeof current.openedAtTranscriptPosition === 'number'
                ? await this.readTranscriptPosition(conversationId)
                : undefined;

        const startedAt =
            typeof metadataConversation.phaseClockStartedAt === 'number'
                ? metadataConversation.phaseClockStartedAt
                : now;
        const banked =
            (typeof metadataConversation.chatElapsedAccumulatedMs === 'number'
                ? metadataConversation.chatElapsedAccumulatedMs
                : 0) + Math.max(0, now - startedAt);

        const updated = await MetadataConversationsModel.findOneAndUpdate(
            {
                _id: new mongoose.Types.ObjectId(conversationId),
                segments: { $elemMatch: { number: current.number, closedAt: null } },
            },
            {
                $set: {
                    'segments.$.closedAt': now,
                    'segments.$.closedBy': reason,
                    'segments.$.closedAtMessageNumber': metadataConversation.messagesNumber ?? 0,
                    ...(closedAtTranscriptPosition !== undefined
                        ? { 'segments.$.closedAtTranscriptPosition': closedAtTranscriptPosition }
                        : {}),
                    chatElapsedAccumulatedMs: banked,
                },
                // Clock stops between blocks; survey pages don't count.
                $unset: { phaseClockStartedAt: '' },
            },
            { new: true },
        ).lean();

        // Lost the race; the other close stands.
        if (!updated) return null;

        // Sync the in-memory copy; it's read again later in the request.
        metadataConversation.segments = (updated as any).segments;
        metadataConversation.chatElapsedAccumulatedMs = banked;
        delete metadataConversation.phaseClockStartedAt;

        return ((updated as any).segments || []).find((s: SegmentRecord) => s.number === current.number) ?? null;
    };

    // Open the next block and restart the chat clock.
    private openNextSegment = async (
        conversationId: string,
        metadataConversation: any,
        record: SegmentRecord,
    ): Promise<SegmentRecord> => {
        const update: any = {
            $push: { segments: record },
            $set: { phaseClockStartedAt: record.openedAt },
            // New block refills the agent-only budget, atomically with the open.
            $unset: { autonomousRoundsUsed: '' },
        };
        // Write-once export origin; phaseClockStartedAt is re-stamped per block.
        if (typeof metadataConversation.chatClockFirstStartedAt !== 'number') {
            update.$set.chatClockFirstStartedAt = record.openedAt;
            metadataConversation.chatClockFirstStartedAt = record.openedAt;
        }
        await this.updateConversationMetadata(conversationId, update);

        metadataConversation.segments = [...(metadataConversation.segments || []), record];
        metadataConversation.phaseClockStartedAt = record.openedAt;
        return record;
    };

    // `segmentsInit` puts the first block in the same insert, so the partial unique index can stop
    // duplicate conversations. Its clock anchors now, so a participant who never types still hits
    // a time quota. The conversation count comes from the user document, never the caller.
    createConversation = async (userId: string, experimentId: string, segmentsInit?: SegmentRecord) => {
        let agent;
        const [user, experiment, experimentBoundries] = await Promise.all([
            usersService.getUserById(userId),
            experimentsService.getExperiment(experimentId),
            experimentsService.getExperimentBoundries(experimentId),
        ]);

        // `?? 0` for users created before the field had a default.
        const userConversationsNumber = user?.numberOfConversations ?? 0;

        // Absent for an unsegmented conversation.
        const segmentFields = segmentsInit
            ? {
                  segments: [segmentsInit],
                  phaseClockStartedAt: segmentsInit.openedAt,
                  chatClockFirstStartedAt: segmentsInit.openedAt,
              }
            : {};

        const isPrivileged = user.role === 'admin' || user.role === 'user';

        // Only participants are capped; admins and researchers are piloting.
        if (
            !isPrivileged &&
            experimentBoundries.maxConversations &&
            userConversationsNumber + 1 > experimentBoundries.maxConversations
        ) {
            throw new AppError('Conversations limit exceeded', 403, 'CONVERSATION_LIMIT_EXCEEDED');
        }

        // Participants carry mode on their user doc; privileged users read it from the experiment.
        const isGroup = isPrivileged
            ? experiment?.agentsMode === AgentsMode.GROUP
            : user.mode === AgentsMode.GROUP;

        if (isGroup) {
            // Privileged users get the live roster; participants get their snapshot.
            const privilegedAgents = isPrivileged ? await experimentsService.getGroupAgents(experimentId) : [];
            const rosterAgents = resolveGroupAgents(user, privilegedAgents, isPrivileged);
            // Pin resolved pacing so DEFAULT_PACING edits don't re-pace running conversations.
            // Split-only agents need humanized: false, since a pacing key alone reads as HD.
            // toObject first: spreading a hydrated Mongoose doc drops its schema fields.
            const pacedAgents = rosterAgents.map((a) => {
                const plain: any = typeof (a as any)?.toObject === 'function' ? (a as any).toObject() : a;
                if (isHumanized(plain)) {
                    return { ...plain, pacing: resolvePacing(plain.pacing), humanized: true };
                }
                if (isSplitEnabled(plain)) {
                    const resolved = resolvePacing(plain.pacing);
                    return {
                        ...plain,
                        pacing: {
                            ...(resolved.enabled ? { enabled: resolved.enabled } : {}),
                            splitProbability: resolved.splitProbability,
                            maxSegments: resolved.maxSegments,
                        },
                        humanized: false,
                    };
                }
                return { ...plain };
            });

            // Merge the experiment's phase schedule onto each agent's phase text in the snapshot,
            // freezing the arc against mid-study edits.
            const agents = pacedAgents.map((a: any) => {
                const { agent: phased, droppedTextPhases } = mergeAgentPhases(a, experiment?.phaseSchedule);
                // Log dropped prompts so it's visible that they never ran.
                if (droppedTextPhases > 0) {
                    console.warn(
                        `[phases] ${a.displayName || a.title} declares ${droppedTextPhases} phase text(s) past the ` +
                            `end of the schedule; a phase with no trigger can never start, so they were dropped.`,
                    );
                }
                return phased;
            });

            const res = await MetadataConversationsModel.create({
                conversationNumber: userConversationsNumber + 1,
                experimentId,
                userId,
                ownerId: experiment?.ownerId,
                agent: agents[0], // placeholder to satisfy the required `agent` field
                agents,
                mode: AgentsMode.GROUP,
                turnTaking: experiment?.turnTaking, // snapshot
                // Seed for reproducible speaker/timing draws.
                seed: crypto.randomBytes(8).toString('hex'),
                maxMessages: isPrivileged ? undefined : experimentBoundries.maxMessages,
                ...segmentFields,
            });

            // Opening sentences. firstMessageSender picks agent1 ([0]), agent2 ([1]) or both.
            const firstMessageSender = experiment?.firstMessageSender ?? 'both';
            const seedAgents =
                firstMessageSender === 'agent1'
                    ? agents.slice(0, 1)
                    : firstMessageSender === 'agent2'
                    ? agents.slice(1, 2)
                    : agents;
            const firstDocs = buildFirstSentenceDocs(seedAgents);
            await Promise.all([
                ...firstDocs.map((doc) =>
                    this.createMessageDoc(doc, res._id.toString(), doc.messageNumber, 0, 0, 0, 0, 0),
                ),
                usersService.addConversation(userId),
                !isPrivileged && experimentsService.addSession(experimentId),
            ]);

            return res._id.toString();
        }

        if (isPrivileged) {
            agent = await experimentsService.getActiveAgent(experimentId);
        }

        const snapAgent = isPrivileged ? agent : user.agent;
        // Same phase merge as the group branch. The seed checks below read the unmerged snapAgent.
        const { agent: phasedAgent, droppedTextPhases } = mergeAgentPhases(snapAgent, experiment?.phaseSchedule);
        if (droppedTextPhases > 0) {
            console.warn(
                `[phases] ${snapAgent?.displayName || snapAgent?.title} declares ${droppedTextPhases} phase text(s) ` +
                    `past the end of the schedule; a phase with no trigger can never start, so they were dropped.`,
            );
        }
        const res = await MetadataConversationsModel.create({
            conversationNumber: userConversationsNumber + 1,
            experimentId,
            userId,
            ownerId: experiment?.ownerId,
            agent: phasedAgent,
            maxMessages: isPrivileged ? undefined : experimentBoundries.maxMessages,
            // Seed only for 1-1 conversations that make seeded draws (HD or split); otherwise none.
            ...(isHumanized(snapAgent) || isSplitEnabled(snapAgent)
                ? { seed: crypto.randomBytes(8).toString('hex') }
                : {}),
            ...segmentFields,
        });

        // An agent may have no opening message; then the participant speaks first.
        const firstChatContent = isPrivileged ? agent.firstChatSentence : user.agent.firstChatSentence;
        await Promise.all([
            ...(firstChatContent
                ? [this.createMessageDoc(
                    { role: 'assistant', content: firstChatContent, timeDelay: null, prev_message: null },
                    res._id.toString(), 1, 0, 0, 0, 0, 0,
                  )]
                : []),
            usersService.addConversation(userId),
            !isPrivileged && experimentsService.addSession(experimentId),
        ]);

        return res._id.toString();
    };

    // Resolve a survey entry into a live chat block. Idempotent, so reloads, back buttons,
    // double clicks and second tabs are harmless.
    // A researcher's preview binds to their own user id so it never provisions a participant.
    enterSegment = async (params: {
        experimentId: string;
        externalPid?: string;
        requestedSegment?: number | null;
        // The authenticated researcher, if any.
        sessionUser?: { _id: string; role: string } | null;
    }): Promise<{
        // Absent when needsRegistration.
        conversationId?: string;
        token?: string;
        segment: ClientSegmentView | null;
        isPreview: boolean;
        exhausted: boolean;
        // No participant for this pid yet; nothing was created. The frame shows registration.
        needsRegistration?: boolean;
        // Requested block deferred because a round is in flight; `segment` is still the old
        // block, and the client should re-enter shortly.
        pendingSegment?: number;
    }> => {
        const { experimentId, externalPid, requestedSegment, sessionUser } = params;
        const experiment = await experimentsService.getExperiment(experimentId);
        if (!experiment) throw new AppError('Experiment not found', 404, 'EXPERIMENT_NOT_FOUND');

        const config: SegmentsConfig | undefined = (experiment as any)?.segments;
        if (!config?.enabled) {
            throw new AppError('This experiment is not survey-embedded', 400, 'SEGMENTS_NOT_ENABLED');
        }

        const isPreview =
            !!sessionUser &&
            (sessionUser.role === 'admin' || sessionUser.role === 'user') &&
            String(experiment.ownerId) === String(sessionUser._id);

        // isActive is checked only at entry, so in-flight participants can finish. Previews are exempt.
        if (!isPreview && !experiment.isActive) {
            throw new AppError('Experiment is not active', 403, 'EXPERIMENT_INACTIVE');
        }

        let user: any;
        if (isPreview) {
            user = await usersService.getUserById(sessionUser!._id);
        } else {
            const pid = (externalPid || '').trim();
            if (!pid) throw new AppError('Missing participant id', 400, 'MISSING_EXTERNAL_PID');
            user = await usersService.getUserByExternalPid(experimentId, pid);
            // Create nothing until the participant registers, or agents could greet a placeholder name.
            if (!user) {
                return { segment: null, isPreview: false, exhausted: false, needsRegistration: true };
            }
        }
        const userId = String(user._id);

        // Segmented only, matching the partial unique index; ignores older unsegmented conversations.
        let meta: any = await MetadataConversationsModel.findOne({
            userId,
            experimentId,
            segments: { $exists: true },
        });

        const now = Date.now();
        const declaredCount = Number(config.count) || 0;

        if (!meta) {
            const first = this.buildSegmentRecord(
                // A first entry always starts at block 1, even if the link names a later one.
                1,
                config,
                now,
                0,
                0,
                // Empty transcript; the opening lines land above 0, inside block 1.
                0,
            );
            const conversationId = await this.createConversation(userId, experimentId, first);
            return {
                conversationId,
                token: isPreview ? undefined : usersService.issueToken(userId),
                segment: clientSegmentView(first, declaredCount, evaluateQuota(first.quota, {
                    userMessagesInSegment: 0,
                    elapsedInSegmentMs: 0,
                }).floorMet),
                isPreview,
                exhausted: false,
            };
        }

        const decision = resolveEntry({
            segments: meta.segments,
            declaredCount,
            requestedSegment,
        });

        if (decision.action === 'closeThenOpen') {
            // Moving forward past an open block. If a round is in flight (including agent-only),
            // defer: record both the close and the block to open, and the round boundary does both.
            // Common, not a rare race: a round is in flight for the whole provider latency.
            if (await this.isRoundInFlight(meta, now)) {
                await this.updateConversationMetadata(
                    meta._id.toString(),
                    {
                        $set: {
                            'segments.$[open].closeRequested': 'override',
                            'segments.$[open].openAfterClose': decision.open,
                        },
                    },
                    { arrayFilters: [{ 'open.closedAt': null }] },
                );
                // The client may still mount on this block, which refills its own budget.
                await this.refillAutonomousBudget(meta._id.toString());
                const current = openSegment(meta.segments);
                return {
                    conversationId: meta._id.toString(),
                    token: isPreview ? undefined : usersService.issueToken(userId),
                    segment: clientSegmentView(current, declaredCount, true, config.endOfSegmentNotice),
                    isPreview,
                    exhausted: false,
                    pendingSegment: decision.open,
                };
            }
            await this.closeOpenSegment(meta._id.toString(), meta, 'override', now);
        }

        if (decision.action === 'exhausted') {
            const last = (meta.segments || []).find((s: SegmentRecord) => s.number === decision.lastSegment);
            return {
                conversationId: meta._id.toString(),
                token: isPreview ? undefined : usersService.issueToken(userId),
                segment: clientSegmentView(last, declaredCount, true, config.endOfSegmentNotice),
                isPreview,
                exhausted: true,
            };
        }

        if (decision.action === 'open' || decision.action === 'closeThenOpen') {
            const target = decision.action === 'open' ? decision.segment : decision.open;
            const [userMessagesSoFar, transcriptPosition] = await Promise.all([
                ConversationsModel.countDocuments({
                    conversationId: meta._id.toString(),
                    role: 'user',
                }),
                this.readTranscriptPosition(meta._id.toString()),
            ]);
            const record = this.buildSegmentRecord(
                target,
                config,
                now,
                meta.messagesNumber ?? 0,
                userMessagesSoFar,
                transcriptPosition,
                decision.skippedFrom,
            );
            await this.openNextSegment(meta._id.toString(), meta, record);
            return {
                conversationId: meta._id.toString(),
                token: isPreview ? undefined : usersService.issueToken(userId),
                segment: clientSegmentView(
                    record,
                    declaredCount,
                    evaluateQuota(record.quota, { userMessagesInSegment: 0, elapsedInSegmentMs: 0 }).floorMet,
                ),
                isPreview,
                exhausted: false,
            };
        }

        // Reuse. Evaluated, not just read, so a time ceiling that passed meanwhile closes the block.
        const segment = await this.evaluateSegmentBoundary(meta._id.toString(), meta, config, now);
        // A remount on an open block refills the client's budget, so refill the server's too.
        if (segment?.isOpen) await this.refillAutonomousBudget(meta._id.toString());
        return {
            conversationId: meta._id.toString(),
            token: isPreview ? undefined : usersService.issueToken(userId),
            segment,
            isPreview,
            exhausted: false,
        };
    };

    // Store one break's survey answers and render them into per-agent knowledge.
    // `gap` is required, never inferred from the open block: the webhook can land after re-entry.
    // Idempotent per gap, since survey platforms retry webhooks.
    recordSurveyAnswers = async (params: {
        experimentId: string;
        externalPid?: string;
        // Preview only (no external pid); ownership is checked by the controller.
        conversationId?: string;
        gap: number;
        values: Record<string, unknown>;
    }): Promise<{ gap: number; accepted: boolean; keys: string[] }> => {
        const { experimentId, externalPid, conversationId, gap, values } = params;
        if (!Number.isFinite(gap) || gap < 1) {
            throw new AppError('A gap number is required', 400, 'INVALID_MESSAGE_FORMAT');
        }
        if (!externalPid && !conversationId) {
            throw new AppError('Missing participant id', 400, 'MISSING_EXTERNAL_PID');
        }

        let meta: any;
        if (conversationId) {
            meta = await MetadataConversationsModel.findOne({
                _id: new mongoose.Types.ObjectId(conversationId),
                segments: { $exists: true },
            });
        } else {
            const user = await usersService.getUserByExternalPid(experimentId, externalPid!);
            if (!user) throw new AppError('Unknown participant', 404, 'USER_NOT_FOUND');
            meta = await MetadataConversationsModel.findOne({
                userId: String(user._id),
                experimentId,
                segments: { $exists: true },
            });
        }
        if (!meta) throw new AppError('No conversation for this participant', 404, 'INVALID_CONVERSATION_ID');

        const key = String(gap);
        if (meta.surveyAnswers?.[key]) {
            return { gap, accepted: false, keys: Object.keys(meta.surveyAnswers[key]?.values || {}) };
        }

        const experiment = await experimentsService.getExperiment(experimentId);
        const config: SegmentsConfig | undefined = (experiment as any)?.segments;

        // Per agent: the agent's own template, else the experiment default.
        const roster: IAgent[] = meta.agents?.length ? meta.agents : [meta.agent];
        const knowledge: Record<string, string[]> = { ...(meta.knowledge || {}) };
        for (const a of roster) {
            if (!a?._id) continue;
            const own = (a as any).knowledgeTemplates?.[gap - 1];
            const template = own ?? config?.defaultKnowledgeTemplates?.[gap - 1];
            const rendered = renderKnowledgeTemplate(template, values);
            if (!rendered) continue;
            const id = String(a._id);
            knowledge[id] = [...(knowledge[id] || []), rendered];
        }

        await this.updateConversationMetadata(meta._id.toString(), {
            $set: {
                [`surveyAnswers.${key}`]: { values: values || {}, receivedAt: Date.now() },
                knowledge,
            },
        });

        return { gap, accepted: true, keys: Object.keys(values || {}) };
    };

    // The single place a block closes on quota; a pure read when no ceiling is reached.
    // Also drains a deferred close and its requested open, since this runs at a round boundary.
    evaluateSegmentBoundary = async (
        conversationId: string,
        meta: any,
        config: SegmentsConfig,
        now = Date.now(),
    ): Promise<ClientSegmentView | null> => {
        const declaredCount = Number(config?.count) || 0;
        const current = openSegment(meta?.segments);

        if (!current) {
            // Report the last block; a closed block is always proceedable so the host's Next shows.
            const last = (meta?.segments || []).reduce(
                (best: SegmentRecord | undefined, s: SegmentRecord) => (!best || s.number > best.number ? s : best),
                undefined as SegmentRecord | undefined,
            );
            return clientSegmentView(last, declaredCount, true, config?.endOfSegmentNotice);
        }

        const userMessagesTotal = await ConversationsModel.countDocuments({ conversationId, role: 'user' });
        const verdict = evaluateQuota(current.quota, {
            userMessagesInSegment: userMessagesInSegment(current, userMessagesTotal),
            elapsedInSegmentMs: Math.max(0, chatElapsedMs(meta, now) - this.segmentStartElapsed(meta, current)),
        });

        // A deferred close outranks the quota: the participant has already moved on.
        const reason: SegmentCloseReason | null =
            (current.closeRequested as SegmentCloseReason | undefined) ??
            (verdict.ceilingHit ? verdict.closedBy : null);

        if (!reason) {
            return clientSegmentView(current, declaredCount, verdict.floorMet, config?.endOfSegmentNotice);
        }

        const closed = await this.closeOpenSegment(conversationId, meta, reason, now);
        // Lost the close race to a concurrent evaluation; report the block as proceedable.
        if (!closed) {
            return clientSegmentView(
                (meta.segments || []).find((s: SegmentRecord) => s.number === current.number),
                declaredCount,
                true,
                config?.endOfSegmentNotice,
            );
        }

        // Open the block a deferred entry asked for, and report it as the current state.
        const opened = await this.drainDeferredOpen(conversationId, meta, config, closed, userMessagesTotal, now);
        if (opened) {
            // No finish check: a forward target means the closed block wasn't the last.
            return clientSegmentView(
                opened,
                declaredCount,
                evaluateQuota(opened.quota, { userMessagesInSegment: 0, elapsedInSegmentMs: 0 }).floorMet,
            );
        }

        if (isFinalSegment(closed.number, declaredCount)) {
            await this.finishSegmentedConversation(conversationId, meta);
        }
        return clientSegmentView(closed, declaredCount, true, config?.endOfSegmentNotice);
    };

    // Open the block a deferred entry asked for, right after its close. Returns null for
    // anything but a clear forward open; the next entry then resolves the state.
    private drainDeferredOpen = async (
        conversationId: string,
        meta: any,
        config: SegmentsConfig,
        closed: SegmentRecord,
        userMessagesTotal: number,
        now: number,
    ): Promise<SegmentRecord | null> => {
        // Read from the record written back by the close, not this request's older copy.
        // Finite check: NaN would pass every comparison below.
        const raw = closed.openAfterClose;
        const target = typeof raw === 'number' && Number.isFinite(raw) ? Math.floor(raw) : null;
        if (target === null) return null;

        const declaredCount = Number(config?.count) || 0;
        // Forward and within the count, re-checked since the flag may be old and the count edited.
        if (target <= closed.number) return null;
        if (declaredCount && target > declaredCount) return null;
        // Never two open blocks; unrecoverable if it happened.
        if (openSegment(meta.segments)) return null;

        const record = this.buildSegmentRecord(
            target,
            config,
            now,
            meta.messagesNumber ?? 0,
            userMessagesTotal,
            // At a round boundary, so the deferring round's turns are all below this.
            await this.readTranscriptPosition(conversationId),
            // Same skip rule as resolveEntry.
            target > closed.number + 1 ? closed.number : undefined,
        );
        await this.openNextSegment(conversationId, meta, record);
        return record;
    };

    // Finish a segmented conversation. isParticipant comes from the bound user, not the caller:
    // the public intake route has no session, and a missed decrement can never be redone.
    // Only the update that flips isFinished decrements the session, so retries are safe.
    private finishSegmentedConversation = async (conversationId: string, meta: any): Promise<void> => {
        const user = await usersService.getUserById(meta.userId);
        const isParticipant = user?.role === 'participant';
        const res = await MetadataConversationsModel.updateOne(
            { _id: new mongoose.Types.ObjectId(conversationId), isFinished: { $ne: true } },
            { $set: { isFinished: true } },
        );
        meta.isFinished = true;
        if (res.modifiedCount && isParticipant) {
            await experimentsService.closeSession(meta.experimentId);
        }
    };

    // Close the open block on a terminal error so the participant can continue the host survey
    // instead of being stuck behind a hidden Next button.
    private closeSegmentOnTerminalError = async (
        conversationId: string,
        metadataConversation: any,
        reason: SegmentCloseReason,
    ): Promise<void> => {
        const current = openSegment(metadataConversation?.segments);
        if (!current) return;
        const experiment = await experimentsService.getExperiment(metadataConversation.experimentId);
        const config: SegmentsConfig | undefined = (experiment as any)?.segments;
        if (!config?.enabled) return;

        const closed = await this.closeOpenSegment(conversationId, metadataConversation, reason);
        if (closed && isFinalSegment(closed.number, Number(config.count) || 0)) {
            await this.finishSegmentedConversation(conversationId, metadataConversation);
        }
    };

    // Pre-turn refusals, shared by 1-1, group and audio. Hitting maxMessages mid-block also
    // closes the block, or the participant would never be released.
    private assertCanSendTurn = async (conversationId: string, metadataConversation: any): Promise<void> => {
        if (isSegmented(metadataConversation?.segments) && !openSegment(metadataConversation.segments)) {
            // Enforced server-side; the client's composer lock can be stale or bypassed.
            throw new AppError('This part of the conversation has ended.', 403, 'SEGMENT_CLOSED');
        }
        if (
            metadataConversation.maxMessages &&
            metadataConversation.messagesNumber + 1 > metadataConversation.maxMessages
        ) {
            await this.closeSegmentOnTerminalError(conversationId, metadataConversation, 'message-limit');
            throw new AppError('Message limit exceeded', 403, 'MESSAGE_LIMIT_EXCEEDED');
        }
    };

    // Runs at the end of every round: the only safe moment to close a block.
    // Re-reads metadata: the caller's copy predates the messagesNumber bump and any deferred
    // close a survey entry wrote during the round.
    private evaluateSegmentAfterRound = async (
        conversationId: string,
        metadataConversation: any,
    ): Promise<ClientSegmentView | null> => {
        if (!isSegmented(metadataConversation?.segments)) return null;
        const experiment = await experimentsService.getExperiment(metadataConversation.experimentId);
        const config: SegmentsConfig | undefined = (experiment as any)?.segments;
        if (!config?.enabled) return null;
        const fresh = await this.getConversationMetadata(conversationId);
        if (!fresh) return null;
        return this.evaluateSegmentBoundary(conversationId, fresh, config);
    };

    // Close the open block from the preview panel, via the same path a quota uses.
    // Recorded as 'manual' so pilots are distinguishable in the export.
    previewCloseSegment = async (conversationId: string): Promise<ClientSegmentView | null> => {
        const meta = await this.getConversationMetadata(conversationId);
        if (!meta || !isSegmented(meta.segments)) return null;
        const experiment = await experimentsService.getExperiment(meta.experimentId);
        const config: SegmentsConfig | undefined = (experiment as any)?.segments;
        if (!config?.enabled) return null;

        const current = openSegment(meta.segments);
        if (!current) return this.evaluateSegmentBoundary(conversationId, meta, config);

        const closed = await this.closeOpenSegment(conversationId, meta, 'manual');
        if (closed && isFinalSegment(closed.number, Number(config.count) || 0)) {
            await this.finishSegmentedConversation(conversationId, meta);
        }
        return clientSegmentView(closed, Number(config.count) || 0, true, config.endOfSegmentNotice);
    };

    // Finish beacon from the host survey: without a quota nothing else closes the last block.
    // A server-side survey-flow call, since an iframe postMessage races the navigation. Idempotent.
    finishSegments = async (params: {
        experimentId: string;
        externalPid?: string;
        // Preview only, as in recordSurveyAnswers.
        conversationId?: string;
    }): Promise<{ finished: boolean; alreadyFinished: boolean; segment: ClientSegmentView | null }> => {
        const { experimentId, externalPid, conversationId } = params;
        if (!experimentId || typeof experimentId !== 'string') {
            throw new AppError('Missing experiment id', 400, 'INVALID_EXPERIMENT_ID');
        }
        if (!externalPid && !conversationId) {
            throw new AppError('Missing participant id', 400, 'MISSING_EXTERNAL_PID');
        }

        let meta: any;
        if (conversationId) {
            meta = await MetadataConversationsModel.findOne({
                _id: new mongoose.Types.ObjectId(conversationId),
                segments: { $exists: true },
            });
        } else {
            const user = await usersService.getUserByExternalPid(experimentId, externalPid!);
            if (!user) throw new AppError('Unknown participant', 404, 'USER_NOT_FOUND');
            meta = await MetadataConversationsModel.findOne({
                userId: String(user._id),
                experimentId,
                segments: { $exists: true },
            });
        }
        if (!meta) throw new AppError('No conversation for this participant', 404, 'INVALID_CONVERSATION_ID');

        const experiment = await experimentsService.getExperiment(meta.experimentId);
        const config: SegmentsConfig | undefined = (experiment as any)?.segments;
        if (!config?.enabled) throw new AppError('Not a survey-embedded study', 400, 'EXPERIMENT_MISCONFIGURED');

        const declaredCount = Number(config.count) || 0;
        const current = openSegment(meta.segments);
        if (!current) {
            // Already closed (beacon ran, or a ceiling hit first): still a success for the host.
            const last = (meta.segments || []).reduce(
                (a: SegmentRecord | null, s: SegmentRecord) => (!a || s.number > a.number ? s : a),
                null,
            );
            return {
                finished: !!meta.isFinished,
                alreadyFinished: true,
                segment: clientSegmentView(last, declaredCount, true, config.endOfSegmentNotice),
            };
        }

        const closed = await this.closeOpenSegment(meta._id.toString(), meta, 'host-finished');

        // Finishes whatever block this was: a survey may legitimately branch past later chat pages.
        if (closed) await this.finishSegmentedConversation(meta._id.toString(), meta);

        return {
            finished: true,
            alreadyFinished: false,
            segment: clientSegmentView(closed, declaredCount, true, config.endOfSegmentNotice),
        };
    };

    // Read (and, where a ceiling has been reached, advance) the current block's state.
    readSegmentState = async (conversationId: string): Promise<ClientSegmentView | null> => {
        const meta = await this.getConversationMetadata(conversationId);
        if (!meta || !isSegmented(meta.segments)) return null;
        const experiment = await experimentsService.getExperiment(meta.experimentId);
        const config: SegmentsConfig | undefined = (experiment as any)?.segments;
        if (!config?.enabled) return null;
        return this.evaluateSegmentBoundary(conversationId, meta, config);
    };

    // Chat-clock value when this block opened, so in-block elapsed is a subtraction, not a second clock.
    private segmentStartElapsed = (meta: any, segment: SegmentRecord): number => {
        const closedBefore = (meta.segments || []).filter(
            (s: SegmentRecord) => s.number < segment.number && s.closedAt != null,
        );
        return closedBefore.reduce(
            (sum: number, s: SegmentRecord) => sum + Math.max(0, (s.closedAt as number) - s.openedAt),
            0,
        );
    };

    // A user round's turn is provisional until the round ends. If the round threw:
    //   - nothing persisted above the user turn: delete it, as if never sent (Resend assumes this);
    //   - anything above it (own reply or another writer's message): keep and count it.
    // Checks the database, not a counter. "Anything", not "an agent turn": deleting under another
    // writer's message leaves a gap that numbering-by-length can never fill.
    // A process that dies mid-round never reaches the finally.
    private withUserRound = async <T>(conversationId: string, run: (round: UserRound) => Promise<T>): Promise<T> => {
        const round: UserRound = { userTurn: null, counted: false };
        try {
            return await run(round);
        } finally {
            // Uncounted here only if the round threw.
            if (round.userTurn && !round.counted) await this.settleUnfinishedUserRound(conversationId, round);
        }
    };

    // The round's single messagesNumber bump. Flagged first so later failures can't take the turn back.
    // Also refills the agent-only budget, as the client does on submit.
    private countUserRound = async (conversationId: string, round: UserRound): Promise<void> => {
        round.counted = true;
        await this.updateConversationMetadata(conversationId, {
            $inc: { messagesNumber: 1 },
            $set: { lastMessageDate: new Date(), lastMessageTimestamp: Date.now() },
            $unset: { autonomousRoundsUsed: '' },
        });
    };

    // Applies the withUserRound rule to a round that threw. Never throws: it must not replace
    // the round's own error.
    private settleUnfinishedUserRound = async (conversationId: string, round: UserRound): Promise<void> => {
        const userTurn = round.userTurn!;
        try {
            // Any role, deliberately (see withUserRound).
            const somethingAbove = await ConversationsModel.exists({
                conversationId,
                messageNumber: { $gt: userTurn.messageNumber },
            });
            if (somethingAbove) {
                // Something above it: removing it would leave a numbering gap.
                await this.countUserRound(conversationId, round);
            } else {
                // Still the last message, so removing it leaves no gap.
                await ConversationsModel.deleteOne({ _id: userTurn._id });
            }
        } catch (error) {
            console.error(
                `[round] conversation ${conversationId}: could not settle the failed round's user turn ` +
                    `#${userTurn.messageNumber} — it may be left unanswered in the transcript: ${error}`,
            );
        }
    };

    // A user round is in flight while user turns in the transcript exceed messagesNumber (the
    // user turn is written first, the count bumped last). Counts user turns, not documents.
    // An agent-only round adds no user turn, so it's detected by its fresh marker instead.
    private isRoundInFlight = async (meta: any, now = Date.now()): Promise<boolean> => {
        if (autonomousRoundMarkerIsFresh(meta.autonomousRoundStartedAt, now)) return true;
        const userTurns = await ConversationsModel.countDocuments({
            conversationId: meta._id.toString(),
            role: 'user',
        });
        return userTurns > (meta.messagesNumber ?? 0);
    };

    getConversation = async (conversationId: string, isLean = false): Promise<Message[]> => {
        const returnValues = isLean
            ? { _id: 0, role: 1, content: 1, timeDelay: 1}
            // Speaker fields exist only on group docs.
            : { _id: 1, role: 1, content: 1, timeDelay: 1, userAnnotation: 1, speakerId: 1, speakerName: 1, messageNumber: 1,
                // Timing envelope, so a reload keeps the server's pacing.
                preTypingDelayMs: 1, typingDurationMs: 1, interSegmentPauseMs: 1,
                segmentGroupId: 1, segmentIndex: 1, segmentCount: 1, revealAtMs: 1, displayedAt: 1,
                // For the client's re-sync. Not in the lean projection, which is sent to the model.
                clientMessageId: 1 };

        const conversation = await ConversationsModel.find({ conversationId }, returnValues).sort({ messageNumber: 1 });

        return conversation;
    };

    getCurrentState = async (conversationId: string) => {
        try {
            console.log("Are you even here?")
            console.log(conversationId);
            const current_state = await CurrentStateModels.find({ id: conversationId }).exec();
            console.log(current_state);
            return current_state;
        } catch (err) {
            console.error(err);
            return null;
        }
    };

    updateConversationSurveysData = async (conversationId: string, data, isPreConversation: boolean) => {
        const saveField = isPreConversation ? { preConversation: data } : { postConversation: data };
        const res = await this.updateConversationMetadata(conversationId, saveField);

        return res;
    };

    getConversationMetadata = async (conversationId: string): Promise<any> => {
        const res = await MetadataConversationsModel.findOne({ _id: new mongoose.Types.ObjectId(conversationId) });
        return res;
    };

    // Participant-facing metadata. The full document holds prompts, model and seed, which would
    // reveal the deception to anyone opening the network tab. Allow-list, so new fields stay hidden.
    getClientConversationMetadata = async (conversationId: string): Promise<any> => {
        const meta = await MetadataConversationsModel.findOne(
            { _id: new mongoose.Types.ObjectId(conversationId) },
            {
                _id: 1,
                experimentId: 1,
                userId: 1,
                mode: 1,
                messagesNumber: 1,
                maxMessages: 1,
                isFinished: 1,
                conversationNumber: 1,
                createdAt: 1,
                'turnTaking.autonomousRounds': 1,
                // For camera/microphone permission prompts.
                'agent.cameraCaptureRate': 1,
                'agent.audioInput': 1,
                // For per-speaker bubble colours.
                'agents._id': 1,
                'agents.displayName': 1,
                'agents.title': 1,
            },
        ).lean();
        return meta;
    };

    getUserConversations = async (userId: string, ccr, vai): Promise<any> => {
        const conversations = [];
        const metadataConversations = await MetadataConversationsModel.find({ userId }, { agent: 0 }).lean();

        for (const metadataConversation of metadataConversations) {
            const conversation = await ConversationsModel.find({
                conversationId: metadataConversation._id,
            }).lean();
            let expAI = {}
            console.log(ccr)
            if (ccr != null && vai != null) {
                expAI = await ExplainableModel.find({
                    conversationId: metadataConversation._id,
                }).lean()
            }
            conversations.push({
                metadata: metadataConversation,
                conversation,
                expAIData: expAI,
            });
        }

        return conversations;
    };

    // The caller's own pilot conversations for one experiment; these are excluded from the
    // participant export. Scoped to the caller's userId so it can't expose participant data.
    getPilotConversations = async (experimentId: string, ownerUserId: string): Promise<any[]> => {
        const metadataConversations = await MetadataConversationsModel.find(
            { experimentId, userId: ownerUserId },
            { agent: 0 },
        )
            .sort({ _id: 1 })
            .lean();

        const conversations = [];
        for (const metadataConversation of metadataConversations) {
            const conversation = await ConversationsModel.find({
                conversationId: metadataConversation._id,
            })
                .sort({ messageNumber: 1 })
                .lean();
            conversations.push({ metadata: metadataConversation, conversation, expAIData: {} });
        }
        return conversations;
    };

    // experimentId is read from the conversation, never the request, so a caller can't decrement
    // another experiment's sessions. The isFinished filter makes the transition atomic.
    finishConversation = async (conversationId: string, isParticipant: boolean): Promise<void> => {
        const finished = await MetadataConversationsModel.findOneAndUpdate(
            { _id: new mongoose.Types.ObjectId(conversationId), isFinished: { $ne: true } },
            { $set: { isFinished: true } },
            { new: true, projection: { experimentId: 1 } },
        ).lean();

        if (finished?.experimentId && isParticipant) {
            await experimentsService.closeSession(String(finished.experimentId));
        }
    };

    deleteExperimentConversations = async (experimentId: string): Promise<void> => {
        const conversationIds = await this.getExperimentConversationsIds(experimentId);
        await Promise.all([
            MetadataConversationsModel.deleteMany({ _id: { $in: conversationIds.ids } }),
            ConversationsModel.deleteMany({ conversationId: { $in: conversationIds.strIds } }),
        ]);
    };

    // A message's conversationId, for the annotation route's ownership check.
    getMessageConversationId = async (messageId: string): Promise<string | null> => {
        // A non-string could cast into a query operator and match someone else's message.
        if (typeof messageId !== 'string' || !mongoose.isValidObjectId(messageId)) return null;
        const message = await ConversationsModel.findById(messageId, { conversationId: 1 }).lean();
        return message?.conversationId ? String(message.conversationId) : null;
    };

    // Scoped by conversationId too, so the write can only land in the ownership-checked conversation.
    updateUserAnnotation = async (
        messageId: string,
        conversationId: string,
        userAnnotation: UserAnnotation,
    ): Promise<Message> => {
        const message: Message = await ConversationsModel.findOneAndUpdate(
            { _id: messageId, conversationId },
            { $set: { userAnnotation } },
            { new: true },
        );

        return message;
    };

    // Client-reported render timestamps, scoped to the caller's conversation. Best-effort.
    reportDisplayedAt = async (
        conversationId: string,
        displays: { messageId: string; displayedAt: number }[],
    ): Promise<void> => {
        // A non-string could cast into a query operator and widen the filter.
        if (typeof conversationId !== 'string' || !conversationId) return;
        if (!Array.isArray(displays) || !displays.length) return;
        const ops = displays
            .slice(0, 100) // bound the write
            // bulkWrite doesn't auto-cast ids, so cast explicitly and drop malformed ones.
            .filter((d) => d && d.messageId && mongoose.isValidObjectId(d.messageId) && Number.isFinite(d.displayedAt))
            .map((d) => ({
                updateOne: {
                    filter: { _id: new mongoose.Types.ObjectId(d.messageId), conversationId },
                    update: { $set: { displayedAt: d.displayedAt } },
                },
            }));
        if (!ops.length) return;
        // Unordered so one failing op doesn't skip the rest.
        await ConversationsModel.bulkWrite(ops, { ordered: false });
    };

    // Always await: the maxMessages guard reads messagesNumber, and an in-flight write lets fast
    // turns run past the cap. Errors are logged, not thrown. `options` carries arrayFilters.
    private updateConversationMetadata = async (conversationId, fields, options?: any) => {
        try {
            const res = await MetadataConversationsModel.updateOne(
                { _id: new mongoose.Types.ObjectId(conversationId) },
                fields,
                options,
            );
            return res;
        } catch (error) {
            console.error(`updateConversationMetadata - ${error}`);
        }
    };

    private processAudio = async (audioBuffer: Buffer) => {
        try {
            const wavBuffer = await convertToWav(audioBuffer);
            return await extractAudioFeatures(wavBuffer);
        } catch (error) {
            console.error("Error processing audio:", error);
            return { pitch: 0, loudness: 0, snr: 0 };
        }
    }


    private transcribeAudio = async (audioBuffer: Buffer, apiKey: string): Promise<string> => {
        const tempFilePath = path.join(os.tmpdir(), `temp_audio_${uuidv4()}.wav`);

        try {
            //console.log(typeof audioBuffer);
            //console.log(audioBuffer);
            //console.log("inside");

            fs.writeFileSync(tempFilePath, audioBuffer);

            const formData = new FormData();
            formData.append("file", fs.createReadStream(tempFilePath));
            formData.append("model", "whisper-1");
            formData.append("language", "en");

            //console.log("Hello");

            const response = await fetch("https://api.openai.com/v1/audio/transcriptions", {
                method: "POST",
                headers: {
                    Authorization: `Bearer ${apiKey}`,
                    ...formData.getHeaders(),
                },
                body: formData as any,
            });

            // console.log("HELLO 2");

            const data = await response.json();
            console.log("Transcription Response:", data);

            return data.text || "";
        } catch (error) {
            console.error("Error transcribing audio:", error);
            return "";
        } finally {
            try {
                fs.unlinkSync(tempFilePath);
            } catch {}
        }
    };

    private transcribeText = async (text: string, apiKey: string): Promise<Buffer> => {
        try {
            const response = await fetch("https://api.openai.com/v1/audio/speech", {
                method: "POST",
                headers: {
                    Authorization: `Bearer ${apiKey}`,
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    model: "tts-1",
                    input: text,
                    voice: "alloy", // also: echo, fable, onyx, nova, shimmer
                }),
            });
    
            if (!response.ok) {
                throw new AppError(`TTS API Error: ${response.statusText}`, 502, 'TTS_PROVIDER_ERROR');
            }
    
            const audioBuffer = Buffer.from(await response.arrayBuffer());
    
            return audioBuffer;
        } catch (error) {
            console.error("Error transcribing text to speech:", error);
            return Buffer.alloc(0);
        }
    };


    private getConversationMessages = (agent: IAgent, conversation: Message[], message: Message, val: number, ar: number, ccr, vai) => {
        const systemPrompt = { role: 'system', content: agent.systemStarterPrompt };
        const beforeUserMessage = { role: 'system', content: agent.beforeUserSentencePrompt };
        const afterUserMessage = { role: 'system', content: agent.afterUserSentencePrompt };
        const inverseTimeDelay = { role: 'system', content: agent.inverseTimeDelay };
        if ( ccr != null && vai != null ) {
            const final_message = "The valence of the user is "+ val + " and the arousal is "+ ar + "while user replies to you " + message["content"]
            message["content"] = final_message
        }
        console.log(message)
        const messages = [
            systemPrompt,
            ...conversation,
            beforeUserMessage,
            message,
            afterUserMessage,
            { role: 'assistant', content: '', timeDelay: inverseTimeDelay },
        ];

        return messages;
    };

    private getExplainableText = (settings: any, conversation: any[], message: any, val: number, ar: number) => {
        const systemPrompt: Message = { role: 'system', content: "", timeDelay: null };
        const beforeUserMessage = { role: 'system', content: "" };
        const afterUserMessage = { role: 'system', content: "" };
        //console.log(message)
        const final_message = "The valence of the user is "+ val + " and the arousal is "+ ar + ". What do you understand from these about the emotions expressed by the user? What behavioral qualities should be displayed while responding to this user to improve their mental state?"
        message["content"] = final_message
        console.log(message)
        const messages: any = [
            systemPrompt,
            ...conversation,
            beforeUserMessage,
            message,
            afterUserMessage,
            { role: 'assistant', content: '' },
        ];

        return messages;
    };

    private createMessageDoc = async (
        message: Message,
        conversationId: string,
        messageNumber: number,
        val: number,
        ar: number,
        pit: number,
        loud: number,
        snr: number,
        // Timing envelope, for group turns and paced 1-1 replies.
        extra?: {
            preTypingDelayMs?: number | null;
            typingDurationMs?: number | null;
            interSegmentPauseMs?: number | null;
            segmentGroupId?: string | null;
            segmentIndex?: number | null;
            segmentCount?: number | null;
            revealAtMs?: number | null;
        },
        // Phase at generation time, not delivery: pacing can hold a bubble past a boundary.
        phase?: { index: number; label?: string | null },
        // Thread-control move ('ground' | 'shift' | 'return'), group only.
        move?: string,
    ): Promise<Message> => {

        // Every message write goes through here, and so through asMessageNumberConflict.
        const res = await ConversationsModel.create({
            content: message.content,
            role: message.role,
            conversationId,
            messageNumber,
            valence: val,
            arousal: ar,
            timeDelay: message.timeDelay,
            prev_message: message.prev_message,
            // Optional keys below are omitted when absent, so other docs keep their shape.
            ...(message.speakerId != null && {
                speakerId: message.speakerId,
                speakerName: message.speakerName,
            }),
            ...(message.clientMessageId != null && { clientMessageId: message.clientMessageId }),
            ...(extra && {
                preTypingDelayMs: extra.preTypingDelayMs,
                typingDurationMs: extra.typingDurationMs,
                interSegmentPauseMs: extra.interSegmentPauseMs,
                segmentGroupId: extra.segmentGroupId,
                segmentIndex: extra.segmentIndex,
                segmentCount: extra.segmentCount,
                revealAtMs: extra.revealAtMs,
            }),
            ...(phase && {
                phaseIndex: phase.index,
                ...(phase.label ? { phaseLabel: phase.label } : {}),
            }),
            ...(move ? { move } : {}),
            pit: pit,
            loud: loud,
            snr: snr,
        }).catch((error) => {
            throw asMessageNumberConflict(error, conversationId, messageNumber);
        });
        //console.log("resTimeDelay", res.timeDelay)
        return {
            _id: res._id,
            role: res.role,
            content: res.content,
            userAnnotation: res.userAnnotation,
            timeDelay: res.timeDelay,
            prev_message: res.prev_message,
            // Without this the envelope is saved but never reaches the browser.
            ...(extra && {
                preTypingDelayMs: res.preTypingDelayMs,
                typingDurationMs: res.typingDurationMs,
                interSegmentPauseMs: res.interSegmentPauseMs,
                segmentGroupId: res.segmentGroupId,
                segmentIndex: res.segmentIndex,
                segmentCount: res.segmentCount,
                revealAtMs: res.revealAtMs,
            }),
            // phaseIndex/phaseLabel/move are deliberately not returned: they would reveal to the
            // participant that the agents are being steered. Only the export reads them.
        };
    };

    private createExplainableDoc = async (og_text: Message, message: Message, resp: Message, conversationId: string, messageNumber: number, val:number, ar:number) => {
        const res = await ExplainableModel.create({
            user_input: og_text.content,
            prompt_input: message.content,
            response: resp.content,
            role: resp.role,
            conversationId,
            messageNumber,
            valence: val, 
            arousal: ar,
        });

        return res;
    };

    private getChatRequest = (agent: IAgent, messages: Message[]) => {
        const chatCompletionsReq = {
            messages,
            model: agent.model,
        };

        // `!= null` so a sampling value of 0 (e.g. temperature 0) is sent, matching the group builders.
        // maxTokens keeps truthiness: 0 means unset.
        if (agent.maxTokens) chatCompletionsReq['max_tokens'] = agent.maxTokens;
        if (agent.frequencyPenalty != null) chatCompletionsReq['frequency_penalty'] = agent.frequencyPenalty;
        if (agent.topP != null) chatCompletionsReq['top_p'] = agent.topP;
        if (agent.temperature != null) chatCompletionsReq['temperature'] = agent.temperature;
        if (agent.presencePenalty != null) chatCompletionsReq['presence_penalty'] = agent.presencePenalty;
        if (agent.cameraCaptureRate) chatCompletionsReq['cameraCaptureRate'] = agent.cameraCaptureRate;
        if (agent.vaIntegration) chatCompletionsReq['vaIntegration'] = agent.vaIntegration;
        if (agent.stopSequences) chatCompletionsReq['stop'] = agent.stopSequences;

        return chatCompletionsReq;
    };

    private getExperimentConversationsIds = async (
        experimentId: string,
    ): Promise<{ ids: mongoose.Types.ObjectId[]; strIds: string[] }> => {
        const conversationsIds = await MetadataConversationsModel.aggregate([
            { $match: { experimentId } },
            { $project: { _id: 1, id: { $toString: '$_id' } } },
            { $group: { _id: null, ids: { $push: '$_id' }, strIds: { $push: '$id' } } },
            { $project: { _id: 0, ids: 1, strIds: 1 } },
        ]);
        return conversationsIds[0];
    };
}

export const conversationsService = new ConversationsService();
