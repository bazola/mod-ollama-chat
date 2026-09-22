-- Deeds a bot has witnessed but not yet digested into memories (plan 41 M4).
--
-- A buffer holding fewer than Memory.EventFlushMinimum deeds is deliberately
-- never digested: flushing near-empty buffers is what took the fleet from ~3.5
-- model calls a minute to ~20, and it is where the quality went, because a
-- model asked to remember one thing pads to fill the quota. That test stays.
--
-- What it cost was the deeds themselves. A bot alone in a dungeon that saw one
-- or two notable things and then logged out lost them entirely, because the
-- buffer lived only in RAM. This table is where it lives instead: written on
-- every save and on logout, reloaded at startup and on login.

CREATE TABLE IF NOT EXISTS mod_ollama_chat_event_buffer (
    bot_guid BIGINT UNSIGNED NOT NULL,
    line_no  INT UNSIGNED NOT NULL COMMENT 'Order within the buffer',
    line     TEXT NOT NULL COMMENT 'Third-person English, place and actor already resolved',
    first_at BIGINT UNSIGNED NOT NULL COMMENT 'Unix seconds the buffer opened; drives the stale timer',
    PRIMARY KEY (bot_guid, line_no)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
