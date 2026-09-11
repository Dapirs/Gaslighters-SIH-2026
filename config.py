# config.py — project-wide constants
 
# Paths
DATA_URL = "https://raw.githubusercontent.com/Vonter/india-mplads-works/main/csv/MPLADS.csv"
RAW_DATA_PATH = "MPLADS.csv"          # used only if you download the CSV locally instead of fetching DATA_URL
SCORED_DATA_PATH = "mplad_scored.csv"  # must match the filename train.py writes and app.py reads
 
# Thresholds
ENTITY_RESOLUTION_THRESHOLD = 85
AGENCY_TYPE_FUZZY_THRESHOLD = 80
AGENCY_PEER_Z_THRESHOLD = 2.0
MIN_PEER_GROUP_SIZE = 5
# NOTE: the four constants above (ENTITY_RESOLUTION_THRESHOLD, AGENCY_TYPE_FUZZY_THRESHOLD,
# AGENCY_PEER_Z_THRESHOLD, MIN_PEER_GROUP_SIZE) are not currently consumed anywhere in
# features.py, graph.py, train.py, or app.py. They look like config for a planned
# agency/contractor entity-resolution module that doesn't exist yet in this codebase.
# Left as-is; wire them up when that module is built, or remove if it's out of scope.
 
# Scoring
SCORABLE_STATUSES = ["Sanctioned", "Ongoing", "Completed"]
# NOTE: also not currently consumed — train.py scores every row regardless of STATUS.
 
# Risk bands
# Expressed on the same 0-1 scale as combined_score in train.py (pd.cut bins:
# [0, RISK_BAND_MEDIUM, RISK_BAND_HIGH, 1.0] -> Low / Medium / High).
# Previously this file had RISK_BAND_HIGH = 70 and RISK_BAND_MEDIUM = 35, which implied
# a 0-100 scale that doesn't match how combined_score is actually computed. Corrected here.
RISK_BAND_MEDIUM = 0.35
RISK_BAND_HIGH = 0.65
 
# IDA fraud-ring graph thresholds (utils/graph.py: build_ida_graph)
# An IDA (implementing agency) is flagged when it spans at least this many
# distinct MPs and states, AND its rejection rate is at least this high.
# NOTE: calibrate MIN_IDA_REJECTION_RATE against the real dataset's overall
# rejection rate before trusting the flagged list — on synthetic test data
# with a 15% baseline rejection rate, ~15% threshold flagged many IDAs by
# chance alone once they spanned 3+ MPs. Check df["ida_rejected"].mean()
# on the real data and set this meaningfully above that baseline.
MIN_IDA_MPS = 3
MIN_IDA_STATES = 2
MIN_IDA_REJECTION_RATE = 0.15
 
# Reproducibility
RANDOM_SEED = 42