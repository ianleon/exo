import os
from pathlib import Path

import anyio
import pytest

os.environ.setdefault("EXO_DASHBOARD_DIR", ".")

from exo.utils.channels import channel
from exo.utils.info_gatherer.info_gatherer import (
    GatheredInfo,
    InfoGatherer,
)
from exo.utils.info_gatherer.macmon import MacmonMetrics

MACMON_SAMPLE = (
    '{"all_power":0.1,"ane_power":0.0,"cpu_power":0.0,'
    '"ecpu_usage":[1000,0.2],"gpu_power":0.0,"gpu_ram_power":0.0,'
    '"gpu_usage":[1000,0.3],"memory":{"ram_total":1000,"ram_usage":250,'
    '"swap_total":100,"swap_usage":10},"pcpu_usage":[1000,0.4],'
    '"ram_power":0.0,"sys_power":1.0,"temp":{"cpu_temp_avg":40.0,'
    '"gpu_temp_avg":41.0},"timestamp":"2026-06-18T00:00:00Z"}'
)


def _write_fake_macmon(path: Path) -> None:
    path.write_text(
        f"""#!/usr/bin/env python3
import sys

print({MACMON_SAMPLE!r}, flush=True)
if "--samples" not in sys.argv:
    print('Error: "Failed to get memory stats"', file=sys.stderr, flush=True)
    raise SystemExit(2)
""",
    )
    path.chmod(path.stat().st_mode | 0o111)


@pytest.mark.asyncio
async def test_monitor_macmon_handles_stream_eof_without_crashing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_macmon = tmp_path / "macmon"
    _write_fake_macmon(fake_macmon)
    monkeypatch.setenv("EXO_MACMON_PATH", os.fsdecode(fake_macmon))

    sender, receiver = channel[GatheredInfo]()
    gatherer = InfoGatherer(sender)

    async with gatherer._tg as task_group:  # pyright: ignore[reportPrivateUsage]
        task_group.start_soon(gatherer._monitor_macmon, 0.01)  # pyright: ignore[reportPrivateUsage]
        with anyio.fail_after(5):
            gathered = await receiver.receive()
        assert isinstance(gathered, MacmonMetrics)

        await anyio.sleep(0.05)
        gatherer.shutdown()
        task_group.cancel_scope.cancel()
