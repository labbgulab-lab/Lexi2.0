import {
    defaultPacing,
    defaultSettings,
    defaultSliderSettings,
    initialSlidersEnabled,
    isAgentDocHumanized,
    isAgentDocSplitting,
} from '@DAL/constants';
import { extractError } from '@DAL/extractError';
import { saveAgent, updateAgent } from '@DAL/server-requests/agents';
import { ChipsInput } from '@components/common/ChipsInput';
import FieldInfo from '@components/common/FieldInfo';
import { SnackbarStatus, useSnackbar } from '@contexts/SnackbarProvider';
import { AgentPacingEnabled, AgentPhase, AgentType } from '@models/AppModels';
import DeleteIcon from '@mui/icons-material/Delete';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import {
    Box,
    Button,
    Checkbox,
    CircularProgress,
    Collapse,
    FormControl,
    FormControlLabel,
    IconButton,
    InputAdornment,
    Slider,
    TextField,
    Tooltip,
    Typography,
} from '@mui/material';

// Slider tooltips by field name; only listed fields get an info icon.
const sliderInfo: Record<string, string> = {
    temperature:
        'How random or creative replies are: near 0 is focused and repeatable, higher is more varied. Leave unset for the model default.',
    maxTokens: 'Upper limit on reply length. Leave unset for the model default.',
    topP: 'An alternative to temperature that restricts word choice to the most likely options; usually adjust one or the other, not both.',
    frequencyPenalty: 'Higher values make the agent less likely to repeat the same words.',
    presencePenalty: 'Higher values push it toward new topics instead of ones already mentioned.',
};
import useActiveUser from '@hooks/useActiveUser';
import { PROMPT_GUIDANCE } from './promptGuidance';
import React, { useEffect, useMemo, useState } from 'react';
import { getExperimentsByAgent } from '../../../../../DAL/server-requests/experiments';
import { updateUsersAgent } from '../../../../../DAL/server-requests/users';
import { WarningMessage } from '../../../../../components/common/WarningMessasge';
import { ModelPicker } from '../../shared/ModelPicker';
import { ModelSettingsNotice } from '../../shared/ModelSettingsNotice';
import { agentUsesConversationPhases } from '../../shared/conversationPhases';
import { MainContainer, SaveButton } from './AgentForm.s';

// Human-Deception sub-toggles in display order. `split` is separate and works with HD off.
const HD_SWITCHES: (keyof AgentPacingEnabled)[] = ['typingSpeed', 'typingVariation', 'preTypingPause'];

// Mirrors the server's knobOn: an HD knob is on unless explicitly false. `split` without a
// flag follows humanization instead, as the server's isSplitEnabled does.
const switchOn = (doc: any, key: keyof AgentPacingEnabled): boolean => {
    const flag = doc?.pacing?.enabled?.[key];
    if (typeof flag === 'boolean') return flag;
    return key === 'split' ? isAgentDocSplitting(doc) : true;
};

// Prompt fields a phase can replace. Not firstChatSentence: it is only sent in phase 1.
const PHASE_PROMPT_FIELDS: { key: keyof AgentPhase; label: string; multiline?: boolean; rows?: number }[] = [
    { key: 'systemStarterPrompt', label: 'System Starter Prompt', multiline: true, rows: 6 },
    { key: 'beforeUserSentencePrompt', label: 'Before User Sentence Prompt' },
    { key: 'afterUserSentencePrompt', label: 'After User Sentence Prompt' },
];

const isBlankPhase = (phase: AgentPhase): boolean => !Object.keys(phase).length;

// Trims and drops blank fields (absent means "inherit the previous phase"). Only trailing blank
// rows are dropped: rows are positional (row i = phase i+2 = boundaries[i]).
const normalizedPhases = (raw: any): AgentPhase[] => {
    const keys: (keyof AgentPhase)[] = ['label', ...PHASE_PROMPT_FIELDS.map((field) => field.key)];
    const rows: AgentPhase[] = (Array.isArray(raw) ? raw : []).map((row: any) => {
        const cleaned: AgentPhase = {};
        keys.forEach((key) => {
            const value = typeof row?.[key] === 'string' ? row[key].trim() : '';
            if (value) cleaned[key] = value;
        });
        return cleaned;
    });
    while (rows.length && isBlankPhase(rows[rows.length - 1])) rows.pop();
    return rows;
};

interface AgentFormProps {
    editAgent: any;
    agents: AgentType[];
    setAgents: (any) => void;
    closeDialog: () => void;
    isEditMode: boolean;
}

const AgentForm: React.FC<AgentFormProps> = ({
    editAgent,
    agents,
    setAgents,
    closeDialog,
    isEditMode = false,
}) => {
    const [isSaveLoading, setIsSaveLoading] = useState(false);
    const [validationMessage, setValidationMessage] = useState('');
    const [experimentsToUpdate, setExperimentsToUpdate] = useState([]);
    const [updateUsersAgentMsg, setUpdateUsersAgentMsg] = useState(false);
    const formTitle = useMemo(() => (!isEditMode ? 'New Agent' : 'Edit Agent'), []);
    const [confirmExperimentUpdateMsg, setConfirmExperimentUpdateMsg] = useState(false);

    const { openSnackbar } = useSnackbar();
    const [slidersEnabled, setSlidersEnabled] = useState<any>(
        isEditMode
            ? {
                  temperatureEnabled: editAgent.temperature !== null,
                  maxTokensEnabled: editAgent.maxTokens !== null,
                  topPEnabled: editAgent.topP !== null,
                  frequencyPenaltyEnabled: editAgent.frequencyPenalty !== null,
                  presencePenaltyEnabled: editAgent.presencePenalty !== null,
                  audioInput: editAgent.audioInput !== null,
              }
            : initialSlidersEnabled,
    );

    const [agent, setAgent] = useState<any>(editAgent ? editAgent : defaultSettings);
    const [isHumanDeception, setIsHumanDeception] = useState(false);
    // Sub-toggles as last left, restored when HD is re-ticked so toggling HD on and off
    // doesn't change the experiment.
    const [hdSwitchMemo, setHdSwitchMemo] = useState<AgentPacingEnabled | null>(null);
    // Open/closed per phase row, positional alongside agent.phases. Seeded to the same length
    // so writes by index land on the right row.
    const [expandedPhases, setExpandedPhases] = useState<boolean[]>(() =>
        Array.isArray(editAgent?.phases) ? editAgent.phases.map(() => false) : [],
    );

    // "Conversation phases" checkbox: derived from content, never stored. Seeded once so
    // deleting the last row doesn't collapse the section mid-edit.
    const [phasesOpen, setPhasesOpen] = useState(() => agentUsesConversationPhases(editAgent));

    // Its own state, not derived from the text, so clearing the field doesn't hide it.
    // New agents send an opening message by default.
    const [sendOpeningMessage, setSendOpeningMessage] = useState(() => !isEditMode || !!editAgent?.firstChatSentence);

    const { activeUser } = useActiveUser();

    // Same predicate the server uses, shared with ExperimentForm.
    useEffect(() => {
        setIsHumanDeception(isAgentDocHumanized(agent));
    }, [agent.humanized, agent.inverseTimeDelay, agent.pacing]);

    // Writes `humanized` explicitly and keeps the pacing block, so switching HD off and on
    // (e.g. to compare with streaming) restores the same settings. humanized: false makes
    // the server ignore the retained fields.
    const handleCheckboxChange = (event) => {
        const checked = event.target.checked;
        setIsHumanDeception(checked);
        if (!checked && isHumanDeception) setHdSwitchMemo(hdSwitchSnapshot());
        const restored = checked ? restoredHdSwitches() : null;
        setAgent((prev) => ({
            ...prev,
            humanized: checked,
            // Off clears inverseTimeDelay: the client fallback (computeMessageDelayMs) still
            // reads it on some 1-1 paths. Pacing has no such reader, so it is kept.
            inverseTimeDelay: checked ? prev.inverseTimeDelay ?? 2.86 : null,
            pacing: checked
                ? { ...(prev.pacing ?? defaultPacing), enabled: { ...prev.pacing?.enabled, ...restored } }
                : prev.pacing ?? null,
        }));
    };

    const hdSwitchSnapshot = (): AgentPacingEnabled =>
        HD_SWITCHES.reduce((acc, key) => ({ ...acc, [key]: switchOn(agent, key) }), {} as AgentPacingEnabled);

    // An all-off memo would immediately untick HD again, so fall back to all on.
    const restoredHdSwitches = (): AgentPacingEnabled => {
        if (hdSwitchMemo && HD_SWITCHES.some((key) => hdSwitchMemo[key])) return hdSwitchMemo;
        return { typingSpeed: true, typingVariation: true, preTypingPause: true };
    };

    // Writes only the touched keys; normalizedAgent fills the rest on save. `humanized` is
    // pinned to the checkbox because a new pacing block alone would read as humanized and
    // tick HD, making split-without-HD impossible. agentPatch can still override it.
    const setPacingSwitches = (patch: AgentPacingEnabled, agentPatch?: any) =>
        setAgent((prev) => ({
            ...prev,
            humanized: isHumanDeception,
            ...agentPatch,
            pacing: {
                ...(prev.pacing ?? defaultPacing),
                enabled: { ...prev.pacing?.enabled, ...patch },
            },
        }));

    const handleHdSwitchChange = (key: keyof AgentPacingEnabled, checked: boolean) => {
        const patch: AgentPacingEnabled = { [key]: checked };
        // Variation only jitters typing time, so it goes off with typing speed.
        if (key === 'typingSpeed' && !checked) patch.typingVariation = false;

        // Without typing speed the engine also drops the reading pause (computeTimingEnvelope),
        // so HD has no effect left and the master box goes off too.
        const rootOff = key === 'typingSpeed' && !checked;
        const stillOn =
            !rootOff && HD_SWITCHES.some((k) => (patch[k] !== undefined ? patch[k] : switchOn(agent, k)));
        if (stillOn || !isHumanDeception) {
            setPacingSwitches(patch);
            return;
        }
        // Nothing humanized is left, so untick HD; snapshot first so re-ticking restores the set.
        setHdSwitchMemo(hdSwitchSnapshot());
        setPacingSwitches(patch, { humanized: false, inverseTimeDelay: null });
    };

    // Keeps the raw string while typing (Number('0.') === 0 would eat a partial decimal);
    // coerced once in normalizedAgent().
    const handlePacingChange = (e) => {
        const { name, value } = e.target;
        setAgent((prev) => ({
            ...prev,
            pacing: { ...(prev.pacing ?? defaultPacing), [name]: value },
        }));
    };

    // Kept as typed; trimmed in normalizedPhases. Never resize here: rows are positional.
    const handlePhaseChange = (index: number, key: keyof AgentPhase, value: string) =>
        setAgent((prev) => {
            const phases = [...(prev.phases ?? [])];
            phases[index] = { ...phases[index], [key]: value };
            return { ...prev, phases };
        });

    // Index taken from phases, not expandedPhases, so the new row opens even if they drift.
    const addPhase = () => {
        const index = (agent.phases ?? []).length;
        setAgent((prev) => ({ ...prev, phases: [...(prev.phases ?? []), {}] }));
        setExpandedPhases((prev) => {
            const next = [...prev];
            next[index] = true;
            return next;
        });
    };

    // normalizedAgent turns an empty list back into an absent key. Flags are filtered at the
    // same index to stay aligned.
    const removePhase = (index: number) => {
        setAgent((prev) => ({ ...prev, phases: (prev.phases ?? []).filter((_, i) => i !== index) }));
        setExpandedPhases((prev) => prev.filter((_, i) => i !== index));
    };

    const togglePhaseExpanded = (index: number) =>
        setExpandedPhases((prev) => {
            const next = [...prev];
            next[index] = !next[index];
            return next;
        });

    // Unticking with content confirms, then clears it: hidden phase text would still run.
    // Set to [] (normalizedAgent's "clear" signal) and reset expandedPhases with the rows.
    const handlePhasesToggle = (event) => {
        if (event.target.checked) {
            setPhasesOpen(true);
            return;
        }
        if (agentUsesConversationPhases(agent)) {
            const ok = window.confirm(
                'Turning off Conversation phases will delete this agent’s phase prompts and its survey ' +
                    'knowledge sentences. Continue?',
            );
            if (!ok) return;
            setAgent((prev) => ({ ...prev, phases: [], knowledgeTemplates: [] }));
            setExpandedPhases([]);
        }
        setPhasesOpen(false);
    };

    // Blank -> undefined (server default); 0 must survive, it disables the pause/split.
    const pacingNumber = (v: unknown): number | undefined => {
        if (v === '' || v === null || v === undefined) return undefined;
        const n = Number(v);
        return Number.isFinite(n) ? n : undefined;
    };

    // The one save boundary: form strings -> stored values.
    const normalizedAgent = (a: any): AgentType => {
        const base: any = { ...a };
        // Unticked keeps the typed text on screen but saves none, so re-ticking restores it.
        if (!sendOpeningMessage) base.firstChatSentence = '';
        // Seed default pacing when HD or splitting is on without a block. isSplitting uses the
        // HD state on screen, since the doc's `humanized` is only updated further down.
        const isSplitting = switchOn({ ...a, humanized: isHumanDeception }, 'split');
        const rawPacing = a.pacing ?? (isHumanDeception || isSplitting ? { ...defaultPacing } : null);
        if (rawPacing == null) {
            // Create: omit the key. Edit: null, so $set clears any saved pacing.
            if (!isEditMode) delete base.pacing;
            else base.pacing = null;
        } else {
            // Lift out `enabled` (an object) before numeric coercion drops it.
            const rawNumbers: any = { ...(rawPacing as any) };
            delete rawNumbers.enabled;
            base.pacing = {
                ...Object.fromEntries(
                    Object.entries(rawNumbers)
                        .map(([k, v]) => [k, pacingNumber(v)])
                        .filter(([, v]) => v !== undefined),
                ),
                // HD switches are written explicitly (same values as the fallback). Split is
                // written only if already explicit: its fallback follows HD, and freezing it
                // would stop a legacy agent splitting after one HD off/on.
                enabled: {
                    typingSpeed: switchOn(a, 'typingSpeed'),
                    typingVariation: switchOn(a, 'typingVariation'),
                    preTypingPause: switchOn(a, 'preTypingPause'),
                    ...(typeof a?.pacing?.enabled?.split === 'boolean' ? { split: isSplitting } : {}),
                },
            };
        }
        // Always explicit: a flagless agent with a pacing block would read as humanized.
        base.humanized = isHumanDeception;
        // See handleCheckboxChange for why this is cleared while pacing is kept.
        if (!isHumanDeception) base.inverseTimeDelay = null;

        // Write phases only when non-empty; [] only to clear phases the loaded doc had
        // ($set leaves omitted keys untouched). Otherwise omit the key.
        const phases = normalizedPhases(a.phases);
        if (phases.length) base.phases = phases;
        else if (isEditMode && Array.isArray(editAgent?.phases) && editAgent.phases.length) base.phases = [];
        else delete base.phases;

        // Same rule as phases.
        const templates = (a.knowledgeTemplates ?? []).map((t: string) => (t || '').trim());
        while (templates.length && !templates[templates.length - 1]) templates.pop();
        if (templates.length) base.knowledgeTemplates = templates;
        else if (isEditMode && Array.isArray(editAgent?.knowledgeTemplates) && editAgent.knowledgeTemplates.length)
            base.knowledgeTemplates = [];
        else delete base.knowledgeTemplates;

        return base;
    };

    const handleTextFieldChange = (e) => {
        setAgent((prev) => ({
            ...prev,
            inverseTimeDelay: e.target.value,
        }));
    };

    const updateAgentInList = (updatedAgent: AgentType) => {
        const updatedSettings = agents.map((agent: AgentType) =>
            agent._id === updatedAgent._id ? updatedAgent : agent,
        );
        setAgents(updatedSettings);
    };

    const validateAgent = (): boolean => {
        let message = '';
        if (!agent.title) message = 'Title is required.';
        else if (!agent.model) message = 'Agent selection is required.';
        else if (sendOpeningMessage && !agent.firstChatSentence?.trim())
            message = 'First Chat Sentence is required when the opening message is on.';
        else if (!agent.systemStarterPrompt) message = 'System Starter Prompt is required.';
        else if (agent.pacing != null) {
            // Report a typo instead of silently falling back to the default. `enabled` holds booleans.
            const bad = Object.entries(agent.pacing).find(
                ([k, v]) =>
                    k !== 'enabled' && v !== '' && v !== null && v !== undefined && !Number.isFinite(Number(v)),
            );
            if (bad) message = `Response pace: "${bad[0]}" must be a number.`;
        }

        // Separate from the chain above: the pacing branch is taken whenever a pacing block exists.
        if (!message) {
            // Trailing blanks are already dropped; a blank row before others wastes a phase.
            const blank = normalizedPhases(agent.phases).findIndex(isBlankPhase);
            if (blank !== -1) {
                message =
                    `Conversation phases: phase ${blank + 2} is empty — it would change nothing. ` +
                    "Fill in a prompt, or give it a label if it is meant to keep the previous phase's text.";
            }
        }

        setValidationMessage(message);
        return !message;
    };

    const handleChange = (event) => {
        const { name, value } = event.target;
        setAgent({ ...agent, [name]: value });
    };

    const handleConfirmUpdate = async () => {
        if (!validateAgent()) return;

        setIsSaveLoading(true);
        try {
            const payload = normalizedAgent(agent);
            await updateAgent(payload);
            updateAgentInList(payload);
            openSnackbar('Agent Updated !', SnackbarStatus.SUCCESS);
            setIsSaveLoading(false);
            setUpdateUsersAgentMsg(true);
            setConfirmExperimentUpdateMsg(false);
        } catch (error) {
            const { message } = extractError(error, 'Agent Update Failed');
            openSnackbar(message, SnackbarStatus.ERROR);
            setIsSaveLoading(false);
        }
    };

    const handleConfirmUsersUpdate = async () => {
        setIsSaveLoading(true);
        try {
            await updateUsersAgent(normalizedAgent(agent));
            openSnackbar('Users Agent Updated !', SnackbarStatus.SUCCESS);
            setIsSaveLoading(false);
            setUpdateUsersAgentMsg(false);
            closeDialog();
        } catch (error) {
            const { message } = extractError(error, 'Users Update Failed');
            openSnackbar(message, SnackbarStatus.ERROR);
            setIsSaveLoading(false);
        }
    };

    const handleUpdate = async () => {
        const experiments = await getExperimentsByAgent(agent._id);
        if (experiments.length) {
            setExperimentsToUpdate(experiments);
            setConfirmExperimentUpdateMsg(true);
            setIsSaveLoading(false);
            return;
        }
        const payload = normalizedAgent(agent);
        await updateAgent(payload);
        updateAgentInList(payload);
        openSnackbar('Agent Updated !', SnackbarStatus.SUCCESS);
        closeDialog();
    };

    const handleSave = async () => {
        if (!validateAgent()) return;

        setIsSaveLoading(true);
        try {
            if (!isEditMode) {
                const savedAgent = await saveAgent(normalizedAgent(agent));
                setAgents([...agents, savedAgent]);
                openSnackbar('Agent Saved !', SnackbarStatus.SUCCESS);
                closeDialog();
            } else {
                await handleUpdate();
            }
            setIsSaveLoading(false);
        } catch (error) {
            const { message } = extractError(error, 'Agent Saving Failed');
            openSnackbar(message, SnackbarStatus.ERROR);
            setIsSaveLoading(false);
        }
    };

    const handleSliderChange = (newValue: number | number[], field: string) => {
        setAgent({ ...agent, [field]: newValue });
    };


    const renderCheckbox = (
        field: string,
        name: string,
        enabled: boolean,
    ) => (
        <FormControl fullWidth margin="normal">
            <Typography gutterBottom>{field.charAt(0).toUpperCase() + field.slice(1)}</Typography>
            <Box display="flex" alignItems="center" gap={1}>
                <Checkbox
                    checked={enabled}
                    onChange={(e) => {
                        handleSliderChange(e.target.checked ? defaultSliderSettings[field] : null, field);
                        setSlidersEnabled({
                            ...slidersEnabled,
                            [`${field}Enabled`]: e.target.checked,
                        });
                    }}
                    name={`${field}Enabled`}
                />
            </Box>
        </FormControl>
    );

    const renderSlider = (
        field: string,
        name: string,
        min: number,
        max: number,
        step: number,
        enabled: boolean,
    ) => (
        <FormControl fullWidth margin="normal">
            <Typography gutterBottom>
                {field.charAt(0).toUpperCase() + field.slice(1)}
                {sliderInfo[field] && <FieldInfo text={sliderInfo[field]} />}
            </Typography>
            <Box display="flex" alignItems="center" gap={1}>
                <Checkbox
                    checked={enabled}
                    onChange={(e) => {
                        handleSliderChange(e.target.checked ? defaultSliderSettings[field] : null, field);
                        setSlidersEnabled({
                            ...slidersEnabled,
                            [`${field}Enabled`]: e.target.checked,
                        });
                    }}
                    name={`${field}Enabled`}
                />
                <Slider
                    value={enabled && agent[field]}
                    onChange={(e, newValue) => handleSliderChange(newValue, field)}
                    min={min}
                    max={max}
                    step={step}
                    valueLabelDisplay="auto"
                    name={name}
                    disabled={!enabled}
                />
            </Box>
        </FormControl>
    );

    return (
        <MainContainer maxWidth="md" style={{ paddingBottom: '32px' }}>
            <Typography variant="h4" gutterBottom margin="normal">
                {formTitle}
            </Typography>
            <TextField
                fullWidth
                required
                label="Title"
                name="title"
                value={agent.title}
                onChange={handleChange}
                size="small"
                margin="normal"
                InputProps={{
                    endAdornment: (
                        <InputAdornment position="end">
                            <FieldInfo text="Internal name for the agent; only researchers see it." />
                        </InputAdornment>
                    ),
                }}
            />
            <TextField
                fullWidth
                label="Display Name"
                name="displayName"
                value={agent.displayName ?? ''}
                onChange={handleChange}
                size="small"
                margin="normal"
                helperText="Shown to participants in chat (group mode); falls back to the title if blank."
                InputProps={{
                    endAdornment: (
                        <InputAdornment position="end">
                            <FieldInfo text="The name participants see for this agent in a group chat (e.g. 'Alex', 'Dr. Lee'). Not shown in single-agent chats." />
                        </InputAdornment>
                    ),
                }}
            />
            <TextField
                maxRows={3}
                rows={3}
                multiline
                fullWidth
                label="Summary"
                name="summary"
                value={agent.summary}
                onChange={handleChange}
                size="small"
                margin="normal"
                InputProps={{
                    endAdornment: (
                        <InputAdornment position="end" sx={{ alignSelf: 'flex-start', mt: 1 }}>
                            <FieldInfo text="Optional note to yourself about the agent's purpose; never shown to participants." />
                        </InputAdornment>
                    ),
                }}
            />
            {/* The owner's model allowance applies, not the editor's (the server enforces it).
                A new agent belongs to the creator. */}
            <Box sx={{ display: 'flex', alignItems: 'flex-start', mt: 2, mb: 1 }}>
                <ModelPicker
                    value={agent.model ?? ''}
                    onChange={(model) => setAgent((prev) => ({ ...prev, model }))}
                    allowedModels={
                        !agent?.ownerId || String(agent.ownerId) === String(activeUser?._id)
                            ? activeUser?.limits?.allowedModels
                            : agent?.ownerAllowedModels
                    }
                />
                <FieldInfo text="Which LLM powers this agent; uses the owner's stored key for that provider. An account restricted to certain models can only be given those." />
            </Box>
            <FormControlLabel
                sx={{ mt: 1 }}
                control={
                    <Checkbox
                        checked={sendOpeningMessage}
                        onChange={(e) => setSendOpeningMessage(e.target.checked)}
                    />
                }
                label="Send automatic opening message"
            />
            {sendOpeningMessage && (
                <TextField
                    fullWidth
                    required
                    multiline
                    maxRows={4}
                    rows={4}
                    label="First Chat Sentence"
                    name="firstChatSentence"
                    value={agent.firstChatSentence ?? ''}
                    onChange={handleChange}
                    size="small"
                    margin="normal"
                    InputProps={{
                        endAdornment: (
                            <InputAdornment position="end" sx={{ alignSelf: 'flex-start', mt: 1 }}>
                                <FieldInfo text="The opening line the agent sends automatically when the conversation starts." />
                            </InputAdornment>
                        ),
                    }}
                />
            )}
            <TextField
                fullWidth
                required
                multiline
                maxRows={8}
                rows={8}
                label="System Starter Prompt"
                name="systemStarterPrompt"
                value={agent.systemStarterPrompt}
                onChange={handleChange}
                size="small"
                margin="normal"
                InputProps={{
                    endAdornment: (
                        <InputAdornment position="end" sx={{ alignSelf: 'flex-start', mt: 1 }}>
                            <FieldInfo text="The agent's core instructions and persona — who it is, how it behaves, what it knows. This is where the real configuration happens." />
                        </InputAdornment>
                    ),
                }}
            />
            <TextField
                fullWidth
                label="Before User Sentence Prompt"
                name="beforeUserSentencePrompt"
                value={agent.beforeUserSentencePrompt}
                onChange={handleChange}
                size="small"
                margin="normal"
                InputProps={{
                    endAdornment: (
                        <InputAdornment position="end">
                            <FieldInfo text="Text quietly inserted right before each participant message, to steer the agent on every turn." />
                        </InputAdornment>
                    ),
                }}
            />
            <TextField
                fullWidth
                label="After User Sentence Prompt"
                name="afterUserSentencePrompt"
                value={agent.afterUserSentencePrompt}
                onChange={handleChange}
                size="small"
                margin="normal"
                InputProps={{
                    endAdornment: (
                        <InputAdornment position="end">
                            <FieldInfo text="Same idea, inserted right after each participant message." />
                        </InputAdornment>
                    ),
                }}
            />

            {/* Always visible, not behind a help icon: researchers won't look for what they don't suspect. */}
            <Box
                sx={{
                    mt: 2,
                    mb: 1,
                    p: 1.5,
                    borderLeft: '3px solid',
                    borderColor: 'divider',
                    backgroundColor: 'rgba(0,0,0,0.02)',
                }}
            >
                {PROMPT_GUIDANCE.map((p) => (
                    <Typography
                        key={p.lead ?? p.body.slice(0, 24)}
                        sx={{ fontSize: '0.75rem', color: 'text.secondary', mb: 0.75, '&:last-of-type': { mb: 0 } }}
                    >
                        {p.lead && <strong>{p.lead} </strong>}
                        {p.body}
                    </Typography>
                ))}
            </Box>

            {/* Hides the phases and survey-knowledge sections for ordinary agents. */}
            <Box sx={{ mt: 2, width: '100%' }}>
                <FormControlLabel
                    control={<Checkbox size="small" checked={phasesOpen} onChange={handlePhasesToggle} />}
                    label={
                        <Tooltip title="Advanced. Lets this agent change character partway through ONE conversation — warm and including at the start, distant and excluding later, for example. You write what it says in each phase here; WHEN each phase begins is set on the experiment. Also holds what the agent learns from a survey, for a study embedded in one. Leave it off for an ordinary agent that keeps one character throughout.">
                            <Typography variant="body2">Conversation phases (advanced)</Typography>
                        </Tooltip>
                    }
                />
            </Box>
            {phasesOpen && (
            <>
            {/* Phase text only; timing is on the experiment so group agents share one clock.
                Phase 1 is the prompts above and is never stored, so row 0 is phase 2. */}
            <Box sx={{ mt: 2, p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
                <Typography>
                    <span>Conversation Phases</span>
                    <FieldInfo text="Lets ONE conversation change character as it goes — warm and including at the start, distant and excluding later, for example. Each phase can replace this agent's three prompts, so every agent in a study still sounds like itself. WHEN each phase begins is set on the experiment, on a single shared schedule, so the agents in a group change together. Leave this empty for an ordinary agent that keeps one character throughout." />
                </Typography>
                <Typography variant="caption" color="text.secondary" component="p">
                    Phase timing is set on the experiment.
                </Typography>

                {/* Read-only: phase 1 is the fields above. */}
                <Box sx={{ mt: 1.5, pl: 1.5, borderLeft: '2px solid', borderColor: 'divider' }}>
                    <Typography variant="subtitle2">Phase 1 — from the start</Typography>
                    <Typography variant="caption" color="text.secondary" component="p">
                        Runs the System Starter / Before / After User Sentence prompts above, from the first
                        message until the next phase begins. Nothing to fill in here.
                    </Typography>
                </Box>

                {(agent.phases ?? []).map((phase: AgentPhase, index: number) => {
                    const setPrompts = PHASE_PROMPT_FIELDS.filter((field) =>
                        (phase?.[field.key] ?? '').trim(),
                    ).length;
                    const isOpen = !!expandedPhases[index];

                    return (
                        <Box
                            key={index}
                            sx={{ mt: 1.5, pl: 1.5, borderLeft: '2px solid', borderColor: 'divider' }}
                        >
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <Typography variant="subtitle2" sx={{ whiteSpace: 'nowrap' }}>
                                    {`Phase ${index + 2}`}
                                </Typography>
                                <TextField
                                    label="Label (optional)"
                                    value={phase?.label ?? ''}
                                    onChange={(e) => handlePhaseChange(index, 'label', e.target.value)}
                                    size="small"
                                    margin="dense"
                                    sx={{ flexGrow: 1 }}
                                    InputProps={{
                                        endAdornment: (
                                            <InputAdornment position="end">
                                                <FieldInfo text="A name for this phase, for your own reading of the exported transcript; participants never see it. If the experiment's schedule names the same phase, that name wins — it is the one every agent in the study shares." />
                                            </InputAdornment>
                                        ),
                                    }}
                                />
                                {/* Prompts collapse behind a count to keep the list scannable. */}
                                <Button
                                    size="small"
                                    onClick={() => togglePhaseExpanded(index)}
                                    endIcon={
                                        <ExpandMoreIcon
                                            sx={{ transform: isOpen ? 'rotate(180deg)' : 'none' }}
                                        />
                                    }
                                >
                                    {isOpen
                                        ? 'Hide prompts'
                                        : `Prompts (${setPrompts}/${PHASE_PROMPT_FIELDS.length})`}
                                </Button>
                                <IconButton
                                    size="small"
                                    aria-label={`Remove phase ${index + 2}`}
                                    onClick={() => removePhase(index)}
                                >
                                    <DeleteIcon sx={{ fontSize: '1.25rem' }} />
                                </IconButton>
                            </Box>
                            <Collapse in={isOpen} unmountOnExit>
                                {PHASE_PROMPT_FIELDS.map((field) => (
                                    <TextField
                                        key={field.key}
                                        fullWidth
                                        multiline={field.multiline}
                                        rows={field.rows}
                                        label={field.label}
                                        value={phase?.[field.key] ?? ''}
                                        onChange={(e) =>
                                            handlePhaseChange(index, field.key, e.target.value)
                                        }
                                        size="small"
                                        margin="dense"
                                        placeholder="(blank = keep the previous phase's)"
                                        // MUI hides the placeholder until the label shrinks;
                                        // shrink up front so "blank = inherit" shows untouched.
                                        InputLabelProps={{ shrink: true }}
                                    />
                                ))}
                            </Collapse>
                        </Box>
                    );
                })}

                <Typography variant="caption" color="text.secondary" component="p" sx={{ mt: 1.5 }}>
                    {(agent.phases?.length ?? 0) === 0
                        ? 'No phases: this agent keeps the prompts above for the whole conversation.'
                        : // How the server handles a length mismatch with the schedule.
                          "Each phase here starts at the matching step of the experiment's schedule. A phase " +
                          'with no step to start it never runs, and a schedule with more steps than this list ' +
                          'keeps the last phase going.'}
                </Typography>
                <Button variant="outlined" size="small" onClick={addPhase} sx={{ mt: 1 }}>
                    Add phase
                </Button>
            </Box>

            {/* Survey knowledge, per agent so group agents can know different things.
                The experiment holds the default. */}
            <Box sx={{ mt: 2 }}>
                <Typography style={{ color: 'grey', marginBottom: '4px', borderBottom: '1px solid grey' }}>
                    <span>What this agent learns from the survey (optional)</span>
                </Typography>
                <Typography variant="caption" color="textSecondary" display="block" sx={{ mb: 1 }}>
                    Only used when the experiment is embedded in a survey. One box per break between chat
                    blocks. Leave blank to use the experiment&apos;s default.
                    <br />
                    Write it as something the agent simply knows — &quot;You know they have been feeling{' '}
                    {'{{mood}}'} lately&quot; — not as a report of a questionnaire, which tells the agent a survey
                    happened and breaks the fiction that this is one continuous conversation. Any sentence whose{' '}
                    {'{{placeholder}}'} went unanswered is dropped rather than left blank.
                    <br />
                    <strong>Keep it narrow.</strong> An agent that reveals something the participant never told it
                    is how they work out the chat is wired to the survey.
                </Typography>
                {/* Always at least one box, even when the list was cleared to []. */}
                {(agent.knowledgeTemplates?.length ? agent.knowledgeTemplates : ['']).map(
                    (template: string, index: number) => (
                    <TextField
                        key={index}
                        fullWidth
                        multiline
                        size="small"
                        label={`After chat block ${index + 1}`}
                        value={template ?? ''}
                        onChange={(e) =>
                            setAgent((prev: any) => {
                                const next = [...(prev.knowledgeTemplates?.length ? prev.knowledgeTemplates : [''])];
                                next[index] = e.target.value;
                                return { ...prev, knowledgeTemplates: next };
                            })
                        }
                        sx={{ mb: 1 }}
                    />
                ))}
                <Button
                    variant="outlined"
                    size="small"
                    onClick={() =>
                        setAgent((prev: any) => ({
                            ...prev,
                            knowledgeTemplates: [
                                ...(prev.knowledgeTemplates?.length ? prev.knowledgeTemplates : ['']),
                                '',
                            ],
                        }))
                    }
                >
                    Add break
                </Button>
            </Box>
            </>
            )}

            <Box>
                {/* Human Deception Checkbox */}
                <FormControlLabel
                    control={
                        <Checkbox
                            checked={isHumanDeception}
                            onChange={handleCheckboxChange}
                            name="humanDeception"
                        />
                    }
                    label={
                        <>
                            <span>Human Deception</span>
                            <FieldInfo text="Makes this agent act human: it takes time to type and pauses to 'read' before replying. Works in BOTH 1-1 and group chats. Leave OFF for an instant, plain reply (like a normal chatbot). Enabling it turns OFF token-by-token streaming for this agent (a typing indicator on top of text that is already appearing contradicts what the participant is watching), and does not apply to audio-input agents. Tick only the parts you want — an unticked part is OFF, not 'use the default'. Blank number fields fall back to the server defaults." />
                        </>
                    }
                />

                {/* Unticked means off; a blank number field below means "use the default". */}
                {isHumanDeception && (
                    <Box sx={{ pl: 4 }}>
                        <FormControlLabel
                            control={
                                <Checkbox
                                    checked={switchOn(agent, 'typingSpeed')}
                                    onChange={(e) => handleHdSwitchChange('typingSpeed', e.target.checked)}
                                    name="pacingTypingSpeed"
                                />
                            }
                            label={
                                <>
                                    <span>Typing speed</span>
                                    <FieldInfo text="The agent takes time to type, showing a typing indicator for as long as the message would take. Turning this OFF also removes the reading pause — a pause followed by a message that appears instantly with no typing indicator is just dead air — and with it nothing in Human Deception has any effect, so unticking this unticks Human Deception itself. Splitting is unaffected: it is its own setting below." />
                                </>
                            }
                        />
                        {switchOn(agent, 'typingSpeed') && (
                            <>
                                <TextField
                                    fullWidth
                                    label="Inverse Time Delay (use 0 for formula presented in Jones C.R. et al. 2024)"
                                    name="inverseTimeDelay"
                                    value={agent.inverseTimeDelay ?? ''}
                                    onChange={handleTextFieldChange}
                                    size="small"
                                    margin="normal"
                                />
                                <TextField
                                    fullWidth
                                    label="Typing Speed (words/sec)"
                                    name="typingSpeed"
                                    value={agent.pacing?.typingSpeed ?? ''}
                                    onChange={handlePacingChange}
                                    size="small"
                                    margin="normal"
                                    InputProps={{
                                        endAdornment: (
                                            <InputAdornment position="end">
                                                <FieldInfo text="Words per second while typing. Blank = use Inverse Time Delay / global default." />
                                            </InputAdornment>
                                        ),
                                    }}
                                />
                            </>
                        )}
                        {/* Disabled without typing speed: the jitter only applies to typing time. */}
                        <Box sx={{ pl: 4 }}>
                            <FormControlLabel
                                disabled={!switchOn(agent, 'typingSpeed')}
                                control={
                                    <Checkbox
                                        checked={
                                            switchOn(agent, 'typingSpeed') && switchOn(agent, 'typingVariation')
                                        }
                                        onChange={(e) => handleHdSwitchChange('typingVariation', e.target.checked)}
                                        name="pacingTypingVariation"
                                    />
                                }
                                label={
                                    <>
                                        <span>Typing variation</span>
                                        <FieldInfo text="Makes each message type at a slightly different speed instead of at a metronome-steady rate. Needs Typing speed — without a typing duration there is nothing to vary." />
                                    </>
                                }
                            />
                            {switchOn(agent, 'typingSpeed') && switchOn(agent, 'typingVariation') && (
                                <TextField
                                    fullWidth
                                    label="Typing Speed Variation"
                                    name="typingSpeedVariation"
                                    value={agent.pacing?.typingSpeedVariation ?? ''}
                                    onChange={handlePacingChange}
                                    size="small"
                                    margin="normal"
                                    InputProps={{
                                        endAdornment: (
                                            <InputAdornment position="end">
                                                <FieldInfo text="Speed jitter (sigma, e.g. 0.22). 0 = every message types at the same rate." />
                                            </InputAdornment>
                                        ),
                                    }}
                                />
                            )}
                        </Box>
                        <FormControlLabel
                            control={
                                <Checkbox
                                    checked={switchOn(agent, 'preTypingPause')}
                                    onChange={(e) => handleHdSwitchChange('preTypingPause', e.target.checked)}
                                    name="pacingPreTypingPause"
                                />
                            }
                            label={
                                <>
                                    <span>Reading pause</span>
                                    <FieldInfo text="The agent waits before it starts typing, as if reading what was just said. The Range below is a +/- on the Mean, so the two are one setting." />
                                </>
                            }
                        />
                        {switchOn(agent, 'preTypingPause') && (
                            <>
                                <TextField
                                    fullWidth
                                    label="Pre-Typing Delay Mean (ms)"
                                    name="preTypingDelayMsMean"
                                    value={agent.pacing?.preTypingDelayMsMean ?? ''}
                                    onChange={handlePacingChange}
                                    size="small"
                                    margin="normal"
                                    InputProps={{
                                        endAdornment: (
                                            <InputAdornment position="end">
                                                <FieldInfo text="Average pause before typing starts, ms (reading the message). 0 = no pause." />
                                            </InputAdornment>
                                        ),
                                    }}
                                />
                                <TextField
                                    fullWidth
                                    label="Pre-Typing Delay Range (ms)"
                                    name="preTypingDelayMsRange"
                                    value={agent.pacing?.preTypingDelayMsRange ?? ''}
                                    onChange={handlePacingChange}
                                    size="small"
                                    margin="normal"
                                    InputProps={{
                                        endAdornment: (
                                            <InputAdornment position="end">
                                                <FieldInfo text="+/- random range on that pause, ms." />
                                            </InputAdornment>
                                        ),
                                    }}
                                />
                            </>
                        )}
                    </Box>
                )}
            </Box>

            {/* Splitting is independent of Human Deception. */}
            <Box>
                <FormControlLabel
                    control={
                        <Checkbox
                            checked={switchOn(agent, 'split')}
                            onChange={(e) => setPacingSwitches({ split: e.target.checked })}
                            name="splitMessages"
                        />
                    }
                    label={
                        <>
                            <span>Split long messages into several</span>
                            <FieldInfo text="A long answer arrives as two or more short messages instead of one wall of text, the way people actually text. Works with or without Human Deception." />
                        </>
                    }
                />
                {switchOn(agent, 'split') && (
                    <>
                        <TextField
                            fullWidth
                            label="Split Probability"
                            name="splitProbability"
                            value={agent.pacing?.splitProbability ?? ''}
                            onChange={handlePacingChange}
                            size="small"
                            margin="normal"
                            InputProps={{
                                endAdornment: (
                                    <InputAdornment position="end">
                                        <FieldInfo text="Chance a long answer is sent as several short messages (0-1). 0 = never split." />
                                    </InputAdornment>
                                ),
                            }}
                        />
                        <TextField
                            fullWidth
                            label="Max Segments"
                            name="maxSegments"
                            value={agent.pacing?.maxSegments ?? ''}
                            onChange={handlePacingChange}
                            size="small"
                            margin="normal"
                            InputProps={{
                                endAdornment: (
                                    <InputAdornment position="end">
                                        <FieldInfo text="How many messages a long answer is split into when it does split. Choose 2-5 (the max is 5; higher values are capped at 5)." />
                                    </InputAdornment>
                                ),
                            }}
                        />
                        <Typography variant="caption" color="text.secondary" component="p">
                            Works with or without Human Deception. A splitting agent cannot stream token-by-token
                            live: it has to hold the whole reply before it can decide where to cut, so with Stream
                            Message on the participant waits a few seconds longer for the first text and then sees
                            each part typed out in turn.
                        </Typography>
                    </>
                )}
            </Box>
            <ModelSettingsNotice model={agent.model} />
            {renderSlider('temperature', 'temperature', 0, 2, 0.01, slidersEnabled.temperatureEnabled)}
            {renderSlider('maxTokens', 'max tokens', 1, 4096, 1, slidersEnabled.maxTokensEnabled)}
            {renderSlider('topP', 'top p', 0, 1, 0.01, slidersEnabled.topPEnabled)}
            {renderSlider(
                'frequencyPenalty',
                'frequency penalty',
                0,
                2,
                0.01,
                slidersEnabled.frequencyPenaltyEnabled,
            )}
            {renderSlider(
                'presencePenalty',
                'presence_penalty',
                0,
                2,
                0.01,
                slidersEnabled.presencePenaltyEnabled,
            )}
            {renderCheckbox('audioInput', 'audioInput', slidersEnabled.audioInputEnabled)}
            <ChipsInput
                list={agent.stopSequences}
                setList={(stops) => setAgent({ ...agent, stopSequences: stops })}
                id={'stop'}
                label={'Stop Sequences'}
                info="Strings that cut the response off if the model is about to produce them."
            />
            {confirmExperimentUpdateMsg ? (
                <WarningMessage
                    handleYes={handleConfirmUpdate}
                    handleNO={() => setConfirmExperimentUpdateMsg(false)}
                >
                    This agent is attached to the following experiments:
                    {experimentsToUpdate.map((experiment, index) => (
                        <Typography
                            key={experiment._id}
                            component="span"
                            display="inline"
                            fontWeight="bold"
                            style={{ marginRight: index < experimentsToUpdate.length - 1 && '8px' }}
                        >
                            {experiment.title}
                            {index < experimentsToUpdate.length - 1 ? ',' : ''}
                        </Typography>
                    ))}
                    . Are you sure you want to update the agent?
                </WarningMessage>
            ) : updateUsersAgentMsg ? (
                <WarningMessage
                    handleYes={handleConfirmUsersUpdate}
                    handleNO={() => {
                        closeDialog();
                        openSnackbar('Agent Saved !', SnackbarStatus.SUCCESS);
                    }}
                >
                    Do you want to update the agent also for users in the experiments that have already been
                    attached to the agent?
                </WarningMessage>
            ) : (
                <SaveButton variant="contained" color="primary" onClick={handleSave} style={{ marginBottom: 0 }}>
                    {isSaveLoading ? <CircularProgress size={28} sx={{ color: 'white' }} /> : 'Save Agent'}
                </SaveButton>
            )}
            {validationMessage && <Typography color="error">{validationMessage}</Typography>}
        </MainContainer>
    );
};

export default AgentForm;
