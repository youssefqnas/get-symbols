# main.py (Gainer-Focused Approach - No Full Sync)
import time
import logging
from datetime import datetime, timezone
from clickhouse_driver import Client as ClickHouseClient
from binance.client import Client as BinanceClient

# --- إعدادات أساسية ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.StreamHandler()
])

# --- بيانات الاتصال ---
CLICKHOUSE_HOST = "l5bxi83or6.eu-central-1.aws.clickhouse.cloud"
CLICKHOUSE_USER = "default"
CLICKHOUSE_PASSWORD = "8aJlVz_A2L4On"
CLICKHOUSE_DB = "default"
BINANCE_API_KEY = ""
BINANCE_API_SECRET = ""

# --- إعدادات السكربت ---
UPDATE_INTERVAL_SECONDS = 120

# --- فلتر جودة البيانات (مهم) ---
MIN_USDT_VOLUME_THRESHOLD = 10000

# --- متغيرات عالمية ---
EPOCH_START_DATETIME = datetime(1970, 1, 1, tzinfo=timezone.utc)

# --- دوال العمليات ---

def get_clickhouse_client():
    try:
        client = ClickHouseClient(host=CLICKHOUSE_HOST, user=CLICKHOUSE_USER, password=CLICKHOUSE_PASSWORD, secure=True, verify=False, database=CLICKHOUSE_DB)
        client.execute('SELECT 1')
        logging.info("Successfully connected to ClickHouse.")
        return client
    except Exception as e:
        logging.error(f"Failed to connect to ClickHouse: {e}")
        return None

def setup_database(client):
    logging.info("Checking and setting up the database table...")
    create_table_query = """
    CREATE TABLE IF NOT EXISTS default.symbols (
        symbol String, registration_date DateTime, host_last_update DateTime,
        guest_last_update DateTime, host_status UInt8, guest_status UInt8,
        is_hot UInt8 DEFAULT 0, hot_rank Int32 DEFAULT 0,

        task_trade UInt8 DEFAULT 0,
        task_trade_last_update DateTime DEFAULT toDateTime(0),

        task_bookTicker UInt8 DEFAULT 0,
        task_bookTicker_last_update DateTime DEFAULT toDateTime(0),

        task_markPrice UInt8 DEFAULT 0,
        task_markPrice_last_update DateTime DEFAULT toDateTime(0),

        task_kline_1m UInt8 DEFAULT 0,
        task_kline_1m_last_update DateTime DEFAULT toDateTime(0),

        task_kline_3m UInt8 DEFAULT 0,
        task_kline_3m_last_update DateTime DEFAULT toDateTime(0),

        task_kline_15m UInt8 DEFAULT 0,
        task_kline_15m_last_update DateTime DEFAULT toDateTime(0),

        task_kline_1h UInt8 DEFAULT 0,
        task_kline_1h_last_update DateTime DEFAULT toDateTime(0)

    ) ENGINE = MergeTree()
    ORDER BY symbol

    """
    try:
        client.execute(create_table_query)
        client.execute("ALTER TABLE default.symbols ADD COLUMN IF NOT EXISTS is_hot UInt8 DEFAULT 0")
        client.execute("ALTER TABLE default.symbols ADD COLUMN IF NOT EXISTS hot_rank Int32 DEFAULT 0")
        logging.info("Table 'symbols' is ready.")
    except Exception as e:
        logging.error(f"Error setting up database table: {e}")
        raise


def manage_gainer_list(ch_client, binance_client):
    """
    تدير قائمة الرابحين: تضيف الجدد، وتحدث الحاليين، وتصفر القدامى.
    """
    logging.info("Fetching 24hr ticker data to manage Gainer list...")
    try:
        all_tickers = binance_client.get_ticker()
    except Exception as e:
        logging.error(f"Failed to fetch ticker data from Binance API: {e}")
        return

    # فلترة وترتيب القائمة
    gainers = [
        t for t in all_tickers
        if t['symbol'].endswith('USDT')
        and float(t['quoteVolume']) > MIN_USDT_VOLUME_THRESHOLD
        and float(t['priceChangePercent']) > 0
    ]
    top_gainers_list = sorted(gainers, key=lambda x: float(x['priceChangePercent']), reverse=True)

    # 1. تصفير حالة كل العملات التي كانت 'hot' سابقاً
    logging.info("Resetting status for previously hot symbols...")
    try:
        ch_client.execute("ALTER TABLE symbols UPDATE is_hot = 0, hot_rank = 0 WHERE is_hot = 1")
    except Exception as e:
        logging.error(f"Failed to reset old hot symbols: {e}")
        return

    if not top_gainers_list:
        logging.warning("No Gainer symbols found in this cycle. All symbols are now marked as not hot.")
        return

    logging.info(f"Identified {len(top_gainers_list)} Gainer symbols. Top 5: {[g['symbol'] for g in top_gainers_list[:5]]}")
    
    # 2. إضافة العملات الرابحة الجديدة التي لا وجود لها في قاعدة البيانات
    db_symbols_set = {row[0] for row in ch_client.execute("SELECT symbol FROM symbols")}
    api_gainers_set = {g['symbol'] for g in top_gainers_list}
    new_symbols_to_insert = api_gainers_set - db_symbols_set

    if new_symbols_to_insert:
        logging.info(f"Found {len(new_symbols_to_insert)} new gainer symbols to insert: {list(new_symbols_to_insert)}")
        now_utc = datetime.now(timezone.utc)
        new_data = []
        # نمر على قائمة الرابحين الأصلية للحصول على الترتيب الصحيح
        for rank, ticker in enumerate(top_gainers_list, 1):
            if ticker['symbol'] in new_symbols_to_insert:
                new_data.append({
                    'symbol': ticker['symbol'], 'registration_date': now_utc, 'host_last_update': now_utc,
                    'guest_last_update': EPOCH_START_DATETIME, 'host_status': 1, 'guest_status': 0,
                    'is_hot': 1, 'hot_rank': rank
                })
        try:
            ch_client.execute("INSERT INTO symbols (symbol, registration_date, host_last_update, guest_last_update, host_status, guest_status, is_hot, hot_rank) VALUES", new_data)
        except Exception as e:
            logging.error(f"Failed to insert new gainer symbols: {e}")

    # 3. تحديث حالة وترتيب كل قائمة الرابحين الحالية (القديمة والجديدة)
    # ملاحظة: هذا الاستعلام سيحدث فقط العملات الموجودة بالفعل في الجدول
    rank_updates = [f"WHEN '{ticker['symbol']}' THEN {rank}" for rank, ticker in enumerate(top_gainers_list, 1)]
    case_statement = " ".join(rank_updates)
    # ندمج تحديث الحالة والترتيب في استعلام واحد
    update_query = f"""
    ALTER TABLE symbols 
    UPDATE is_hot = 1, hot_rank = CASE symbol {case_statement} ELSE hot_rank END
    WHERE symbol IN {tuple(api_gainers_set)}
    """
    try:
        ch_client.execute(update_query)
        logging.info(f"Successfully updated status and rank for {len(api_gainers_set)} gainer symbols.")
    except Exception as e:
        logging.error(f"Failed to update status for gainer symbols: {e}")


def main():
    ch_client = None
    while not ch_client:
        ch_client = get_clickhouse_client()
        if not ch_client:
            logging.warning("Database connection failed, retrying in 30 seconds...")
            time.sleep(30)
            
    binance_client = BinanceClient(BINANCE_API_KEY, BINANCE_API_SECRET)
    setup_database(ch_client)

    while True:
        cycle_start_time = time.time()
        
        # لا يوجد مزامنة كاملة، فقط إدارة قائمة الرابحين
        logging.info("================== [GAINER LIST MANAGEMENT] Starting cycle ==================")
        manage_gainer_list(ch_client, binance_client)
        
        elapsed_time = time.time() - cycle_start_time
        sleep_time = max(0, UPDATE_INTERVAL_SECONDS - elapsed_time)
        logging.info(f"Cycle finished in {elapsed_time:.2f}s. Sleeping for {sleep_time:.2f}s. ==================")
        time.sleep(sleep_time)

if __name__ == "__main__":
    main()