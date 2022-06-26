


BASE_PATH = 'C:\\Users\\Nicholas\\Documents\\workspace\\stock_general'
SAVED_MODELS_PATH = 'C:\\Users\\Nicholas\\Documents\\workspace\\stock_general\\ml\\saved_models'

# common df columns
TIME = 'time'
OPEN = 'open'
HIGH = 'high'
LOW = 'low'
CLOSE = 'close'

# intervals and time_periods
MINUTE_15 = '15min'
MINUTE_30 = '30min'

MONTH_1 = '1month'
MONTH_3 = '3month'
MONTH_5 = '5month'

# price action related terms
LFILTER = 'lfilter'
LFILTER_5 = 'lfilter_5'
LFILTER_10 = 'lfilter_10'
LINREG_15 = 'lfilter_15'

# technical indicators
TIME_NORM = 'time_norm'     # normalized time values --> thrown around a cosine

EMA_5 = 'ema_5'
EMA_20 = 'ema_20'
EMA_50 = 'ema_50'


UPPER_BBAND_STD1 = "upper_bband_std1"
UPPER_BBAND_STD2 = "upper_bband_std2"
LOWER_BBAND_STD1 = "lower_bband_std1"
LOWER_BBAND_STD2 = "lower_bband_std2"

BBAND_BASE_UPPER = 'upper_bband_std'
BBAND_BASE_LOWER = 'lower_bband_std'

TI_BBAND_UP = 'upper_bband'
TI_BBAND_LOW = 'lower_bband'


VOLUME = 'volume'
MARKET_CAP = 'market_cap'
RSI = 'rsi'                 # relative  strength index
VWAP = 'vwap'               # volume weighted average_price
MACD = 'macd'               # moving average convergence / divergence
STOCH = 'stoch'             # stochastic oscillator
STOCHF = 'stoch'            # stochastic fast values
STOCH_RSI = 'stoch_rsi'     # stochastic relative strength index
ADX = 'adx'                 # average directional movement index

HOLDING_STOCK = 3
NO_POSITION = 2
BUY_STOCK = 1
DO_NOTHING = 0
SELL_STOCK = -1
SHORT_BUY_STOCK = -2
SHORT_SELL_STOCK = -3

HOLDING_UPPER_BB = 1
NO_HOLD = 0
HOLDING_LOWER_BB = -1

# machine learning
BATCH_SIZE = 16