"""Cognitive memory layer — the agent's "brain".

Public surface:
- schemas         : data models (Emotion, AffectiveState, Assertion, …, ParticipantContext)
- prosody        : ProsodyProfile, _PROFILES, to_elevenlabs_voice() (Dynamic Prosody)
- pre_call       : build_precall_context() (Dynamic Prosody + Adaptive Verbosity + Proactive Empathy)
- post_call      : process_post_call() / schedule_post_call()           [Step 3]
- graph_engine   : CognitiveGraph (Neo4j async)                          [Step 2]
- acoustic_engine: analyze_audio() (librosa primary + SenseVoice rich)   [Step 3]
"""
