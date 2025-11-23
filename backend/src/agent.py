import logging
import os
import json
import datetime
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a friendly Starbucks barista. The user will place a drink order by speaking.
            Your job is to ask concise clarifying questions until the following order state is fully filled:
            {
              "drinkType": "string",
              "size": "string",
              "milk": "string",
              "extras": ["string"],
              "name": "string"
            }
            Ask one clear question at a time to obtain missing fields. Confirm the full order once complete, then save the order to disk and tell the user it was saved.
            Keep a friendly, helpful tone and avoid complex formatting.""",
        )

        # Initialize a small order state that we will fill via tools
        self.order_state = {
            "drinkType": "",
            "size": "",
            "milk": "",
            "extras": [],
            "name": "",
        }

    @function_tool
    async def update_order(self, context: RunContext, field: str, value: str):
        """Update a specific field of the order.

        Args:
            field: One of 'drinkType', 'size', 'milk', 'extras', 'name'
            value: The value to set. For 'extras', a comma-separated string will be split into list items.

        Returns:
            A short status string describing what changed and any remaining missing fields, or a saved-file confirmation when complete.
        """
        allowed = {"drinkType", "size", "milk", "extras", "name"}
        if field not in allowed:
            return f"Unknown field '{field}'. Allowed fields: {', '.join(sorted(allowed))}."

        if field == "extras":
            extras = [e.strip() for e in value.split(",") if e.strip()]
            if extras:
                # extend list without duplicates
                for e in extras:
                    if e not in self.order_state["extras"]:
                        self.order_state["extras"].append(e)
        else:
            self.order_state[field] = value.strip()

        # Determine missing fields (extras must have at least one item)
        missing = [f for f in ["drinkType", "size", "milk", "extras", "name"] if not self.order_state.get(f)]
        # If extras is empty list, ensure it's considered missing
        if isinstance(self.order_state.get("extras"), list) and not self.order_state.get("extras"):
            if "extras" not in missing:
                missing.append("extras")

        if not missing:
            # Order complete: save to JSON file under backend/backend/orders
            try:
                # Path resolution:
                # this file is at <...>/backend/src/agent.py
                # parents[1] => <...>/backend
                src_parent_backend = Path(__file__).resolve().parents[1]
                orders_dir = src_parent_backend / "backend" / "orders"
                orders_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.exception("Failed to create orders directory")
                return f"Order complete but failed to create orders directory: {e}"

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = orders_dir / f"order_{timestamp}.json"
            tmp_path = filename.with_suffix(".json.tmp")

            try:
                # write atomically to temp then replace
                with tmp_path.open("w", encoding="utf-8") as f:
                    json.dump(self.order_state, f, ensure_ascii=False, indent=2)
                tmp_path.replace(filename)
                logger.info(f"Saved order to {filename}")
            except Exception as e:
                logger.exception(f"Failed to save order to {filename}")
                # attempt to remove tmp file if exists
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass
                return f"Order complete but failed to save: {e}"

            return f"Order complete and saved to {filename}."

        return f"Updated '{field}'. Missing fields: {', '.join(missing)}."

    @function_tool
    async def get_order(self, context: RunContext):
        """Return the current order state as JSON-serializable dict."""
        return self.order_state


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging context
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
