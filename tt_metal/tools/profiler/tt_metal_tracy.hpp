#if defined(TRACY_ENABLE)

#define TracyTTMetalBeginTrace( device_id, trace_id )                                                                 \
    std::string trace_message = fmt::format("`TT_DNN_TRACE_BEGIN: {}, {}`", device_id, trace_id);                     \
    TracyMessage(trace_message.c_str(), trace_message.size());

#define TracyTTMetalEndTrace( device_id, trace_id )                                                                   \
    std::string trace_message = fmt::format("`TT_DNN_TRACE_END: {}, {}`", device_id, trace_id);                       \
    TracyMessage(trace_message.c_str(), trace_message.size());

#define TracyTTMetalReplayTrace( device_id, trace_id )                                                                 \
    std::string trace_message = fmt::format("`TT_DNN_TRACE_REPLAY: {}, {}`", device_id, trace_id);                     \
    TracyMessage(trace_message.c_str(), trace_message.size());

#define TracyTTMetalReleaseTrace( device_id, trace_id )                                                                   \
    std::string trace_message = fmt::format("`TT_DNN_TRACE_RELEASE: {}, {}`", device_id, trace_id);                       \
    TracyMessage(trace_message.c_str(), trace_message.size());
#else

#define TracyTTMetalBeginTrace( device_id, trace_id )
#define TracyTTMetalEndTrace( device_id, trace_id )
#define TracyTTMetalReplayTrace( device_id, trace_id )
#define TracyTTMetalRelkaseTrace( device_id, trace_id )

#endif
