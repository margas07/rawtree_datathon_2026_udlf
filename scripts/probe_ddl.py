"""One-off probe: can our API key CREATE TABLE via the Query API?
Writes ddl_probe_report.txt. Launch detached:
    Start-Process py -ArgumentList '-X','utf8','scripts/probe_ddl.py'
"""
from pathlib import Path
import sys, time, traceback

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.client import RawTreeClient

OUT = ROOT / "ddl_probe_report.txt"
lines = []
def log(msg=""):
    print(msg)
    lines.append(msg)

def main():
    ok = False
    try:
        client = RawTreeClient()
        variants = [
            "CREATE TABLE IF NOT EXISTS experiments ENGINE = ReplicatedRawMergeTree",
            "CREATE TABLE IF NOT EXISTS team_12.experiments ENGINE = ReplicatedRawMergeTree",
        ]
        for sql in variants:
            try:
                client.query(sql)
                log(f"OK   : {sql}")
            except Exception as exc:
                log(f"FAIL : {sql}")
                log(f"       {exc}")
        time.sleep(2)
        try:
            rows = client.query(
                "SELECT database, name, engine FROM system.tables "
                "WHERE database IN ('team_12', currentDatabase())")
            log(f"tables visible: {rows}")
        except Exception as exc:
            log(f"system.tables failed: {exc}")
        try:
            client.insert("experiments", [{"kind": "probe", "name": "ddl_probe",
                                          "metrics": {"ping": 1}, "params": {},
                                          "tags": ["probe"], "ts": "2026-11-14T18:00:00+00:00"}])
            log("insert into experiments: OK")
            rows = client.query("SELECT count() AS n FROM team_12.experiments")
            log(f"read back count: {rows}")
            ok = bool(rows) and int(rows[0].get("n") or 0) >= 1
        except Exception as exc:
            log(f"insert/read-back failed: {exc}")
    except Exception:
        log(traceback.format_exc())
    log(f"=== DDL probe {'PASS' if ok else 'INCONCLUSIVE/FAIL'} ===")
    OUT.write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__":
    main()
