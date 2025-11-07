MindWanderingPrediction

This project investigates mind-wandering by extracting and combining eye-tracking and behavioral features around probe onsets during a sustained attention task.

For each probe, a 10-second window preceding the probe is used to compute:

Eye-tracking features (fixation metrics, saccade dynamics, blink activity, pupil measures)

Behavioral features (tapping variability, derived complexity metrics, BV/ApEn, etc.)

Both feature sets are aligned and merged using:

subj_orgID, session_id, block_num, probe_number


so that each probe corresponds to a single, unified feature representation.

Structure of the Project

The extracted data is used in two distinct modeling pipelines, depending on the research question:

1. ML_PIPELINE (Across-Participant Model)

This pipeline treats all probes across participants as part of a shared feature space.
The goal is to investigate population-level patterns:

What features generally relate to mind-wandering across people?

Can a model trained across subjects generalize?

This pipeline reflects group-level inference.

2. ML_WITHINPARTICIPANT (Subject-Specific Model)

This pipeline runs separate machine learning analyses for each participant.
The goal is to evaluate whether mind-wandering can be predicted reliably within an individual, independent of group patterns.

This pipeline supports:

Within-participant cross-validation

Permutation-based significance testing

Comparison of individual predictability profiles

This reflects person-specific dynamics of mind-wandering.


CSV DATA EXTRACTION:
                ┌─────────────────────────────┐
                │   Raw Eye-Tracking Data      │
                │   (per subject/session)      │
                └──────────────┬───────────────┘
                               │
                               ▼
                    Extract All Annotations
               (fixations / saccades / blinks / pupil)
                               │
                               ▼
           ┌───────────────────────────────────────────────┐
           │                 ProbesVF Indexed               │
           │  (subj_orgID, session_id, block_num, probe#)  │
           └───────────────────────┬───────────────────────┘
                                   │
                                   ▼
                       Align probe onsets (keys)
                                   │
            ┌──────────────────────┴──────────────────────┐
            │                                             │
            ▼                                             ▼
  10s Pre-Probe Window (Eye Data)           10s Pre-Probe Window (Behavioral Data)
            │                                             │
            ▼                                             ▼
   Eye Feature Extraction                         Behavioral Feature Extraction
            │                                             │
            ▼                                             ▼
   Eye Feature Table (per probe)                 Behavioral Feature Table (per probe)
            │                                             │
            └──────────────────────┬──────────────────────┘
                                   ▼
                              Merge on keys:
                    subj_orgID / session_id / block_num / probe_number
                                   │
                                   ▼
                ┌────────────────────────────────┐
                │     Final Merged Feature CSV    │
                │   (one row per probe event)     │
                └────────────────────────────────┘
