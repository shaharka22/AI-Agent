from tools.gdacs_tool     import get_gdacs_alerts,                TOOL_DEFINITION as GDACS_DEF
from tools.usgs_tool      import get_earthquakes_global,           TOOL_DEFINITION as USGS_DEF
from tools.nasa_tool      import get_nasa_events,                  TOOL_DEFINITION as NASA_DEF
from tools.rag_tool       import search_historical_and_analyze,    TOOL_DEFINITION as RAG_DEF
from tools.analytics_tool import query_disaster_statistics,        TOOL_DEFINITION as STATS_DEF
from tools.emergency_tool import get_israel_emergency_contacts,    TOOL_DEFINITION as EMERGENCY_DEF
from tools.telegram_tool  import send_emergency_alert,             TOOL_DEFINITION as TELEGRAM_DEF
from tools.distance_tool  import get_nearest_fire_station,         TOOL_DEFINITION as DISTANCE_DEF

TOOL_REGISTRY = {
    "get_gdacs_alerts":               get_gdacs_alerts,
    "get_earthquakes_global":         get_earthquakes_global,
    "get_nasa_events":                get_nasa_events,
    "search_historical_and_analyze":  search_historical_and_analyze,
    "query_disaster_statistics":      query_disaster_statistics,
    "get_israel_emergency_contacts":  get_israel_emergency_contacts,
    "send_emergency_alert":           send_emergency_alert,
    "get_nearest_fire_station":       get_nearest_fire_station,
}

ALL_TOOLS = [GDACS_DEF, USGS_DEF, NASA_DEF, RAG_DEF, STATS_DEF, EMERGENCY_DEF, TELEGRAM_DEF, DISTANCE_DEF]
