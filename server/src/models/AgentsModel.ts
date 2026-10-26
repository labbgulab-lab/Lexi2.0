import { Schema } from 'mongoose';
import { mongoDbProvider } from '../mongoDBProvider';
import { ConversationPhase, IAgent } from '../types';

// One phase: prompt text from the agent plus timing merged in from the experiment
// (mergeAgentPhases). Timing fields are absent on the agent doc itself.
// `_id: false` keeps embedded snapshots small and comparable.
export const conversationPhaseSchema = new Schema<ConversationPhase>(
    {
        label: { type: String, required: false },
        startAfterSeconds: { type: Number, required: false },
        startAfterUserMessages: { type: Number, required: false },
        // Must also be declared here, or mongoose drops it from the merged snapshot and the phase never fires.
        startAtSegment: { type: Number, required: false },
        systemStarterPrompt: { type: String, required: false },
        beforeUserSentencePrompt: { type: String, required: false },
        afterUserSentencePrompt: { type: String, required: false },
    },
    { _id: false, versionKey: false },
);

export const agentsSchema = new Schema<IAgent>(
    {
        ownerId: { type: String, index: true },
        title: { type: String, required: true },
        displayName: { type: String, required: false },
        summary: { type: String, required: false },
        systemStarterPrompt: { type: String, required: false },
        beforeUserSentencePrompt: { type: String, required: false },
        afterUserSentencePrompt: { type: String, required: false },
        inverseTimeDelay: { type: Number, required: false},
        firstChatSentence: { type: String, required: false },
        model: { type: String, required: true },
        temperature: { type: Number },
        maxTokens: { type: Number },
        topP: { type: Number },
        frequencyPenalty: { type: Number },
        presencePenalty: { type: Number },
        cameraCaptureRate: {type: Number },
        audioInput: {type: Number},
        vaIntegration: { type: Number },
        stopSequences: { type: [String] },
        pacing: {
            typingSpeed: { type: Number, required: false },
            typingSpeedVariation: { type: Number, required: false },
            preTypingDelayMsMean: { type: Number, required: false },
            preTypingDelayMsRange: { type: Number, required: false },
            splitProbability: { type: Number, required: false },
            maxSegments: { type: Number, required: false },
            // Per-knob humanization switches. Absent means enabled; only `false` disables.
            // No defaults, since a default would be written onto every doc mongoose touches.
            enabled: {
                typingSpeed: { type: Boolean, required: false },
                typingVariation: { type: Boolean, required: false },
                preTypingPause: { type: Boolean, required: false },
                split: { type: Boolean, required: false },
            },
        },
        // Humanization on/off. Absent = off.
        humanized: { type: Boolean, required: false },
        // Phase prompt text. `default: undefined` so mongoose doesn't write `[]` onto every doc.
        phases: { type: [conversationPhaseSchema], required: false, default: undefined },
        // Knowledge template per survey gap; index 0 = after block 1.
        knowledgeTemplates: { type: [String], required: false, default: undefined },
        createdAt: { type: Date, default: Date.now },
        timestamp: { type: Number, default: () => Date.now() },
    },
    { versionKey: false },
);

export const AgentsModel = mongoDbProvider.getModel('agents', agentsSchema);
