# config.py — project-wide constants

# Paths
RAW_DATA_PATH = "raw_data.csv"
SCORED_DATA_PATH = "scored_data.csv"
AGENCY_FLAGS_PATH = "agency_flags.csv"

# Thresholds
ENTITY_RESOLUTION_THRESHOLD = 85
AGENCY_TYPE_FUZZY_THRESHOLD = 80
AGENCY_PEER_Z_THRESHOLD = 2.0
MIN_PEER_GROUP_SIZE = 5

# Scoring
SCORABLE_STATUSES = ["Sanctioned", "Ongoing", "Completed"]

# Risk bands
RISK_BAND_HIGH = 70
RISK_BAND_MEDIUM = 35

# Reproducibility
RANDOM_SEED = 42
