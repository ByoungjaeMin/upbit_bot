"""
main.py — 업비트 퀀트 자동매매 봇 진입점

실행: python upbit_bot/main.py
     (또는 프로젝트 루트 ~/quant에서 실행)

[초기화 순서]
  1. .env 로드 및 환경변수 검증
  2. SQLite DB 연결 및 스키마 확인
  3. StorageManager 초기화
  4. WebSocket + 데이터 수집기 연결
  5. 텔레그램 봇 초기화
  6. DRY_RUN 여부 콘솔 출력
  7. 메인 트레이딩 루프 시작

[구조]
  asyncio.gather로 병렬 실행:
    - _main_schedule_loop()  : 5분 주기 trading + 1분 position + 10초 circuit
    - _telegram_loop()       : 텔레그램 봇 폴링
    - collector.start()      : WebSocket 데이터 수집
    - _storage_heartbeat()   : StorageManager 시간별 디스크 점검

  SIGINT / SIGTERM → graceful shutdown
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backtest.monte_carlo import MonteCarloResult

# ─────────────────────────────────────────────────────────────────
# 경로 설정: upbit_bot/ 을 패키지 루트로 등록
# ─────────────────────────────────────────────────────────────────

_BOT_DIR = Path(__file__).parent          # upbit_bot/
_PROJECT_DIR = _BOT_DIR.parent            # ~/quant/

if str(_BOT_DIR) not in sys.path:
    sys.path.insert(0, str(_BOT_DIR))

# ─────────────────────────────────────────────────────────────────
# 로깅 설정 (임포트 전에 먼저)
# ─────────────────────────────────────────────────────────────────

_LOG_DIR = _PROJECT_DIR / "logs"
_LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            _LOG_DIR / f"bot_{datetime.now(timezone.utc).strftime('%Y%m%d')}.log",
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("main")

# ─────────────────────────────────────────────────────────────────
# 1. .env 로드
# ─────────────────────────────────────────────────────────────────

try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_DIR / ".env")
    logger.info("[main] .env 로드 완료")
except ImportError:
    logger.warning("[main] python-dotenv 미설치 — 환경변수를 시스템에서 직접 읽습니다")

# ─────────────────────────────────────────────────────────────────
# 환경변수 수집
# ─────────────────────────────────────────────────────────────────

def _load_yaml_config() -> dict:
    """config.yaml 로드. 파일 없거나 파싱 실패 시 빈 dict 반환."""
    import yaml  # PyYAML — requirements에 포함
    config_path = _BOT_DIR / "config.yaml"
    if not config_path.exists():
        logger.warning("[main] config.yaml 없음 — 코드 기본값 사용")
        return {}
    try:
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        logger.error("[main] config.yaml 파싱 실패: %s — 코드 기본값 사용", exc)
        return {}


def _require_env(key: str) -> str:
    val = os.getenv(key, "")
    if not val:
        logger.error("[main] 필수 환경변수 미설정: %s", key)
        sys.exit(1)
    return val


def _load_initial_capital() -> float:
    """INITIAL_CAPITAL 환경변수 파싱. 미설정 시 경고 후 기본값 1천만원 사용."""
    raw = os.getenv("INITIAL_CAPITAL", "")
    if not raw:
        logger.warning(
            "[main] INITIAL_CAPITAL 미설정 — 기본값 10,000,000원 사용. "
            "실거래 시 .env에 INITIAL_CAPITAL을 명시하라."
        )
        return 10_000_000.0
    try:
        return float(raw)
    except ValueError:
        logger.error("[main] INITIAL_CAPITAL 값 파싱 실패: '%s' — 기본값 10,000,000원 사용", raw)
        return 10_000_000.0


def _load_config() -> dict:
    """환경변수 검증 및 설정 딕셔너리 반환."""
    # TELEGRAM_CHAT_ID / CHAT_ID 양쪽 모두 지원 (.env 호환)
    chat_id_raw = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID", "")
    if not chat_id_raw:
        logger.error("[main] TELEGRAM_CHAT_ID 또는 CHAT_ID 미설정")
        sys.exit(1)

    return {
        "access_key":     _require_env("UPBIT_ACCESS_KEY"),
        "secret_key":     _require_env("UPBIT_SECRET_KEY"),
        "telegram_token": _require_env("TELEGRAM_TOKEN"),
        "telegram_chat_id": int(chat_id_raw),
        "bok_api_key":    os.getenv("BOK_API_KEY", ""),
        "cryptoquant_key": os.getenv("CRYPTOQUANT_KEY", ""),
        "dry_run":        os.getenv("DRY_RUN", "true").lower() not in ("false", "0", "no"),
        "initial_capital": _load_initial_capital(),
        "db_path":        _BOT_DIR / os.getenv("DB_PATH", "data/bot.db"),
        "phase_b":        os.getenv("PHASE_B", "false").lower() in ("true", "1"),
        "phase_c":        os.getenv("PHASE_C", "false").lower() in ("true", "1"),
        "yaml":           _load_yaml_config(),  # config.yaml 전체 (engine에서 사용)
    }


# ─────────────────────────────────────────────────────────────────
# 루프 주기 상수
# ─────────────────────────────────────────────────────────────────

MAIN_LOOP_INTERVAL_SEC     = 300   # 5분 — 신호 → 주문 파이프라인
POSITION_LOOP_INTERVAL_SEC = 60    # 1분 — 포지션 모니터 (트레일링/익절/손절)
CIRCUIT_LOOP_INTERVAL_SEC  = 10    # 10초 — 서킷브레이커 감시
DISK_CHECK_INTERVAL_SEC    = 3600  # 1시간 — 디스크 사용량 점검
SHUTDOWN_TELEGRAM_WAIT_SEC = 3     # 종료 텔레그램 전송 대기 (초)


# ─────────────────────────────────────────────────────────────────
# 봇 애플리케이션
# ─────────────────────────────────────────────────────────────────

class BotApplication:
    """전체 봇 라이프사이클 관리."""

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._shutdown_event = asyncio.Event()

        # 컴포넌트 (run()에서 초기화)
        self._collector = None
        self._engine = None
        self._telegram = None
        self._storage = None
        self._layer1 = None
        self._layer2 = None
        self._selector = None
        self._decay_monitor = None
        self._scheduler = None   # APScheduler AsyncIOScheduler
        self._db_conn = None     # SQLite 연결 (텔레그램 명령어 조회용)

    # ──────────────────────────────────────────────────────────────
    # 초기화 단계
    # ──────────────────────────────────────────────────────────────

    def _step2_init_db(self) -> None:
        """2단계: SQLite DB 연결 및 스키마 초기화."""
        import sqlite3
        from data.cache import init_db
        db_path = self._cfg["db_path"]
        db_path.parent.mkdir(parents=True, exist_ok=True)
        init_db(db_path)
        # 텔레그램 명령어 조회용 읽기 전용 연결 (check_same_thread=False)
        self._db_conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db_conn.row_factory = sqlite3.Row
        logger.info("[main] DB 초기화 완료: %s", db_path)

    def _step3_init_storage(self) -> None:
        """3단계: StorageManager 초기화."""
        from monitoring.storage_manager import StorageManager
        self._storage = StorageManager(
            db_path=self._cfg["db_path"],
            base_dir=_BOT_DIR,
        )
        logger.info("[main] StorageManager 초기화 완료")

    def _step4_init_collector(self) -> None:
        """4단계: UpbitDataCollector (WebSocket + REST) 초기화."""
        from data.collector import UpbitDataCollector
        self._collector = UpbitDataCollector(
            access_key=self._cfg["access_key"],
            secret_key=self._cfg["secret_key"],
            db_path=str(self._cfg["db_path"]),
            bok_api_key=self._cfg["bok_api_key"],
            cryptoquant_key=self._cfg["cryptoquant_key"],
            yaml_config=self._cfg.get("yaml", {}),
        )
        logger.info("[main] UpbitDataCollector 초기화 완료")

    def _step5_init_telegram(self) -> None:
        """5단계: 텔레그램 봇 초기화."""
        from monitoring.telegram_bot import TelegramBot, BotContext
        self._telegram = TelegramBot(
            token=self._cfg["telegram_token"],
            chat_id=self._cfg["telegram_chat_id"],
        )
        logger.info("[main] TelegramBot 초기화 완료")

    def _step6_init_engine(self) -> None:
        """6단계: 레이어 + 엔진 초기화."""
        from execution.order import UpbitClient
        from execution.engine import TradingEngine
        from layers.layer1_filter import Layer1MarketFilter
        from layers.layer2_ensemble import Layer2Ensemble
        from strategies.selector import StrategySelector
        from strategies.decay_monitor import StrategyDecayMonitor
        from risk.circuit_breaker import CircuitBreaker

        cfg = self._cfg

        # 업비트 클라이언트
        client = UpbitClient(
            access_key=cfg["access_key"],
            secret_key=cfg["secret_key"],
            dry_run=cfg["dry_run"],
        )

        # 엔진
        self._engine = TradingEngine(
            upbit_client=client,
            dry_run=cfg["dry_run"],
            initial_capital=cfg["initial_capital"],
            yaml_config=cfg.get("yaml", {}),
        )
        # Phase 자동 전환 감지용 DB 경로 주입 (7가지 체크리스트 조회에 필요)
        self._engine.set_db_path(str(cfg["db_path"]))

        # 레이어 컴포넌트
        self._decay_monitor = StrategyDecayMonitor()
        self._layer1 = Layer1MarketFilter(
            circuit_breaker=self._engine.circuit_breaker,
            phase_c_enabled=cfg["phase_c"],
        )
        self._layer2 = Layer2Ensemble(
            trade_count_fn=lambda: self._engine.trade_count,
            phase_b_enabled=cfg["phase_b"],
        )
        self._selector = StrategySelector(
            decay_monitor=self._decay_monitor,
            phase_c_enabled=cfg["phase_c"],
        )

        self._engine.setup_layers(
            layer1=self._layer1,
            layer2=self._layer2,
            strategy_selector=self._selector,
            decay_monitor=self._decay_monitor,
        )

        # StorageManager 텔레그램 알림 연결
        if self._storage and self._telegram:
            self._storage.set_telegram_callback(self._telegram.send)

        logger.info("[main] TradingEngine + 레이어 초기화 완료")

    def _step7_set_bot_context(self) -> None:
        """7단계: 텔레그램 봇 컨텍스트 연결 + 수집기에 서킷브레이커 주입."""
        # 수집기에 서킷브레이커 후(後) 주입 — step4(collector)보다 step6(engine)이 늦게 초기화됨
        if self._collector is not None and self._engine is not None:
            self._collector.set_circuit_breaker(self._engine.circuit_breaker)

        from monitoring.telegram_bot import BotContext
        ctx = BotContext(
            engine=self._engine,
            paper_runner=self._engine._paper,
            circuit_breaker=self._engine.circuit_breaker,
            decay_monitor=self._decay_monitor,
            layer1=self._layer1,
            layer2=self._layer2,
            db_conn=self._db_conn,
            current_phase="Phase B" if self._cfg["phase_b"] else "Phase A",
        )
        self._telegram.set_context(ctx)
        # 200건 달성 알림 콜백 주입 (set_context 이후 — bot이 준비된 상태에서 주입)
        self._engine.set_telegram_callback(self._telegram.send)

    # ──────────────────────────────────────────────────────────────
    # APScheduler 초기화 및 스케줄 작업
    # ──────────────────────────────────────────────────────────────

    def _init_scheduler(self) -> None:
        """APScheduler AsyncIOScheduler 초기화 및 크론 작업 등록."""
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            from apscheduler.triggers.cron import CronTrigger
        except ImportError:
            logger.warning("[main] apscheduler 미설치 — 스케줄 작업 비활성화")
            return

        self._scheduler = AsyncIOScheduler(timezone="UTC")

        # coin_history 스냅샷은 scripts/coin_history_collector.py 독립 프로세스가 담당.
        # 중복 실행 방지를 위해 아래 job 등록을 비활성화함.
        # self._scheduler.add_job(
        #     self._job_coin_history_snapshot,
        #     CronTrigger(hour=0, minute=0),
        #     id="coin_history_snapshot",
        #     name="코인 히스토리 스냅샷",
        #     misfire_grace_time=300,
        # )

        # 매일 01:00 UTC — DataQuality 리포트 + 텔레그램
        self._scheduler.add_job(
            self._job_data_quality_report,
            CronTrigger(hour=1, minute=0),
            id="data_quality_report",
            name="데이터 품질 리포트",
            misfire_grace_time=300,
        )

        # 매일 14:00 UTC (= 23:00 KST) — 일간 텔레그램 리포트
        self._scheduler.add_job(
            self._job_daily_report,
            CronTrigger(hour=14, minute=0),
            id="daily_report",
            name="일간 텔레그램 리포트",
            misfire_grace_time=300,
        )

        # 매주 일요일 05:00 UTC — StrategyDecayMonitor 주간 통계
        self._scheduler.add_job(
            self._job_decay_weekly,
            CronTrigger(day_of_week="sun", hour=5, minute=0),
            id="decay_weekly",
            name="전략 Decay 주간 통계",
            misfire_grace_time=600,
        )

        # 매 4시간 — GridStrategy 범위 재계산
        self._scheduler.add_job(
            self._job_grid_recalc,
            CronTrigger(minute=0, hour="*/4"),
            id="grid_recalc",
            name="그리드 범위 재계산",
            misfire_grace_time=120,
        )

        # 매 5분 — DCA Safety Order 모니터링
        self._scheduler.add_job(
            self._job_dca_safety_check,
            CronTrigger(minute="*/5"),
            id="dca_safety_check",
            name="DCA Safety Order 모니터링",
            misfire_grace_time=60,
        )

        # 매일 19:00 UTC (= 04:00 KST 다음날) — 페어리스트 갱신
        self._scheduler.add_job(
            self._job_pairlist_refresh,
            CronTrigger(hour=19, minute=0),
            id="pairlist_refresh",
            name="페어리스트 갱신",
            misfire_grace_time=300,
        )

        # 매주 일요일 02:00 UTC — Walk-Forward 백테스트 + Monte Carlo 검증
        self._scheduler.add_job(
            self._job_weekly_backtest,
            CronTrigger(day_of_week="sun", hour=2, minute=0),
            id="weekly_backtest",
            name="주간 Walk-Forward + Monte Carlo 백테스트",
            misfire_grace_time=3600,
        )

        logger.info("[main] APScheduler 7개 크론 작업 등록 완료 (coin_history는 독립 프로세스)")

    async def _job_coin_history_snapshot(self) -> None:
        """매일 00:00 UTC — 업비트 KRW 마켓 코인 목록 스냅샷 저장."""
        logger.info("[scheduler] coin_history 스냅샷 시작")
        try:
            if self._collector and hasattr(self._collector, "snapshot_coin_history"):
                await self._collector.snapshot_coin_history()
            elif self._db_conn and self._collector:
                # 기본 구현: 현재 페어리스트를 coin_history에 저장
                from datetime import datetime, timezone
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                pairs = self._collector.pairlist.get_active_pairs() if self._collector else []
                for i, coin in enumerate(pairs, 1):
                    self._db_conn.execute(
                        "INSERT OR IGNORE INTO coin_history"
                        " (snapshot_date, coin, rank, included_in_pairlist)"
                        " VALUES (?, ?, ?, 1)",
                        (today, coin, i),
                    )
                self._db_conn.commit()
                logger.info("[scheduler] coin_history 저장 완료 (%d개)", len(pairs))
        except Exception as exc:
            logger.error("[scheduler] coin_history 스냅샷 오류: %s", exc)

    async def _job_data_quality_report(self) -> None:
        """매일 01:00 UTC — DataQuality 점검 리포트 → 텔레그램."""
        logger.info("[scheduler] DataQuality 리포트 시작")
        try:
            if not self._db_conn or not self._telegram:
                return
            row = self._db_conn.execute(
                "SELECT COUNT(*) as total, SUM(tradeable) as ok"
                " FROM layer1_log"
                " WHERE timestamp >= datetime('now', '-24 hours')"
            ).fetchone()
            total = row["total"] or 0
            ok = row["ok"] or 0
            rate = ok / total if total else 0
            candle_row = self._db_conn.execute(
                "SELECT COUNT(DISTINCT coin) as coins FROM candles_5m"
                " WHERE timestamp >= datetime('now', '-10 minutes')"
            ).fetchone()
            active = candle_row["coins"] if candle_row else 0
            msg = (
                f"✅ <b>일일 데이터 품질 리포트</b>\n"
                f"Layer1 24h: {total}건 (Tradeable {rate:.0%})\n"
                f"활성 코인: {active}개"
            )
            await self._telegram.send(msg, priority=2)
        except Exception as exc:
            logger.error("[scheduler] DataQuality 리포트 오류: %s", exc)

    async def _job_daily_report(self) -> None:
        """매일 14:00 UTC (23:00 KST) — 일간 텔레그램 리포트."""
        logger.info("[scheduler] 일간 리포트 시작")
        try:
            if not self._db_conn or not self._telegram:
                return
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            summary = self._db_conn.execute(
                "SELECT COUNT(*) as cnt,"
                "       SUM(CASE WHEN side='SELL' THEN pnl ELSE 0 END) as pnl,"
                "       AVG(CASE WHEN side='SELL' AND pnl>0 THEN 1.0 ELSE 0.0 END) as wr"
                " FROM trades"
                " WHERE timestamp >= ? AND is_dry_run=0",
                (today,),
            ).fetchone()
            strat_rows = self._db_conn.execute(
                "SELECT strategy_type, SUM(pnl)/MAX(krw_amount)*100 as contrib_pct"
                " FROM trades"
                " WHERE timestamp >= ? AND side='SELL' AND is_dry_run=0"
                " GROUP BY strategy_type",
                (today,),
            ).fetchall()
            import shutil
            disk = shutil.disk_usage("/")
            free_gb = disk.free / 1024 ** 3

            engine_status = self._engine.get_status() if self._engine else {}
            capital = engine_status.get("capital", 0)

            paper_row = self._db_conn.execute(
                "SELECT AVG(CAST(signal_match AS FLOAT)) as match_rate"
                " FROM paper_comparison WHERE week_start >= ?",
                (today,),
            ).fetchone()
            paper_match = paper_row["match_rate"] or 0 if paper_row else 0

            strat_contrib = {
                r["strategy_type"]: (r["contrib_pct"] or 0)
                for r in strat_rows
            }
            from monitoring.telegram_bot import MessageFormatter
            msg = MessageFormatter.daily_report(
                total_trades=summary["cnt"] or 0,
                realized_pnl=summary["pnl"] or 0,
                win_rate=summary["wr"] or 0,
                capital=capital,
                strategy_contrib=strat_contrib,
                paper_match_rate=paper_match,
                disk_free_gb=free_gb,
            )
            await self._telegram.send(msg, priority=2)
        except Exception as exc:
            logger.error("[scheduler] 일간 리포트 오류: %s", exc)

    async def _job_decay_weekly(self) -> None:
        """매주 일요일 05:00 UTC — StrategyDecayMonitor 주간 통계."""
        logger.info("[scheduler] 전략 Decay 주간 통계 시작")
        try:
            if self._decay_monitor and hasattr(self._decay_monitor, "update_weekly_stats"):
                conn = self._engine._db_conn if self._engine else None
                if conn:
                    await asyncio.to_thread(
                        self._decay_monitor.update_weekly_stats, conn
                    )
                    logger.info("[scheduler] 전략 Decay 주간 통계 완료")
        except Exception as exc:
            logger.error("[scheduler] Decay 주간 통계 오류: %s", exc)

    async def _job_grid_recalc(self) -> None:
        """매 4시간 — GridStrategy 범위 재계산."""
        logger.info("[scheduler] 그리드 범위 재계산 시작")
        try:
            if not self._engine:
                return
            positions = getattr(self._engine, "positions", {})
            grid_coins = [
                coin for coin, pos in positions.items()
                if getattr(pos, "strategy_type", "") == "GRID"
            ]
            if not grid_coins:
                return
            if hasattr(self._engine, "_recalc_grid_ranges"):
                market_states = await self._collect_market_states()
                await self._engine._recalc_grid_ranges(grid_coins, market_states)
                logger.info("[scheduler] 그리드 재계산 완료 (%d개)", len(grid_coins))
        except Exception as exc:
            logger.error("[scheduler] 그리드 재계산 오류: %s", exc)

    async def _job_dca_safety_check(self) -> None:
        """매 5분 — DCA Safety Order 모니터링."""
        try:
            if not self._engine:
                return
            positions = getattr(self._engine, "positions", {})
            dca_coins = [
                coin for coin, pos in positions.items()
                if getattr(pos, "strategy_type", "") == "DCA"
            ]
            if not dca_coins:
                return
            if hasattr(self._engine, "_check_dca_safety_orders"):
                market_states = await self._collect_market_states()
                await self._engine._check_dca_safety_orders(dca_coins, market_states)
        except Exception as exc:
            logger.error("[scheduler] DCA Safety Order 오류: %s", exc)

    async def _job_pairlist_refresh(self) -> None:
        """매일 19:00 UTC (= 04:00 KST) — 페어리스트 갱신."""
        logger.info("[scheduler] 페어리스트 갱신 시작")
        try:
            if self._collector and hasattr(self._collector, "refresh_pairlist"):
                await self._collector.refresh_pairlist()
                logger.info("[scheduler] 페어리스트 갱신 완료")
        except Exception as exc:
            logger.error("[scheduler] 페어리스트 갱신 오류: %s", exc)

    async def _job_weekly_backtest(self) -> None:
        """매주 일요일 02:00 UTC — Walk-Forward 백테스트 + Monte Carlo 검증.

        순서:
          1. scripts/run_walk_forward.py subprocess 실행
          2. trades 테이블에서 OOS pnl 시퀀스 로드
          3. MonteCarloValidator.validate() 실행
          4. backtest_results 테이블 저장
          5. 결과 텔레그램 알림 (실패 시 priority=1 긴급)
        """
        logger.info("[scheduler] 주간 백테스트 시작")

        script_path = _BOT_DIR / "scripts" / "run_walk_forward.py"
        db_path     = str(self._cfg["db_path"])
        output_dir  = _PROJECT_DIR / "results" / "walk_forward"
        run_id      = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # ── 1. run_walk_forward.py subprocess ─────────────────────
        wf_ok = False
        try:
            end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            proc = await asyncio.create_subprocess_exec(
                sys.executable, str(script_path),
                "--coins",      "BTC,ETH,XRP,SOL,ADA",
                "--start",      "2022-01-01",
                "--end",        end_date,
                "--output-dir", str(output_dir),
                "--db-path",    db_path,
                "--no-telegram",    # 텔레그램 알림은 이 job에서 직접 전송
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=7200  # 최대 2시간
            )
            if proc.returncode == 0:
                logger.info("[scheduler] Walk-Forward subprocess 완료")
                wf_ok = True
            else:
                err_tail = (
                    stderr_bytes.decode("utf-8", errors="replace")[-500:]
                    if stderr_bytes else ""
                )
                logger.error(
                    "[scheduler] Walk-Forward 실패 (rc=%d): %s",
                    proc.returncode, err_tail,
                )

        except asyncio.TimeoutError:
            logger.error("[scheduler] Walk-Forward 타임아웃 (2시간 초과)")
            if self._telegram:
                await self._telegram.send(
                    "🚨 <b>Walk-Forward 타임아웃</b> — 2시간 초과\n수동 확인 필요",
                    priority=1,
                )
            return

        except Exception as exc:
            logger.error("[scheduler] Walk-Forward subprocess 오류: %s", exc)
            if self._telegram:
                await self._telegram.send(
                    f"🚨 <b>Walk-Forward subprocess 오류</b>\n{exc}",
                    priority=1,
                )
            return

        if not wf_ok:
            if self._telegram:
                await self._telegram.send(
                    "🚨 <b>Walk-Forward 실패</b>\n"
                    "데이터 부족 또는 Lookahead Bias 오염 감지\n"
                    "logs/run_walk_forward.log 확인 필요",
                    priority=1,
                )
            return

        # ── 2~4. Monte Carlo 검증 + 저장 ──────────────────────────
        try:
            pnl_pcts = await asyncio.to_thread(self._load_oos_pnls, db_path)

            from backtest.monte_carlo import MonteCarloValidator
            validator  = MonteCarloValidator()
            mc_result  = await asyncio.to_thread(validator.validate, pnl_pcts)

            logger.info("[scheduler] %s", mc_result.summary())

            await asyncio.to_thread(
                self._save_backtest_results_sqlite,
                mc_result, run_id, db_path,
            )

            # ── 5. 텔레그램 알림 ──────────────────────────────────
            if self._telegram:
                status = "✅ PASS" if mc_result.passed else "🚨 FAIL"
                p_label = "< 0.05 ✓" if mc_result.passed else ">= 0.05 ✗"
                msg = (
                    f"<b>📊 주간 백테스트 완료</b> {status}\n"
                    f"Monte Carlo p-value: {mc_result.p_value:.4f} ({p_label})\n"
                    f"실제 샤프: {mc_result.actual_sharpe:.3f}\n"
                    f"셔플 평균 샤프: {mc_result.shuffle_sharpe_mean:.3f}\n"
                    f"엣지 신뢰도: {mc_result.edge_confidence:.1%}\n"
                    f"MDD: {mc_result.actual_max_drawdown:.1%}\n"
                    f"95% CI: [{mc_result.sharpe_ci_lower:.3f}, {mc_result.sharpe_ci_upper:.3f}]\n"
                    f"거래 수: {mc_result.n_trades}건"
                )
                priority = 1 if not mc_result.passed else 2
                await self._telegram.send(msg, priority=priority)

            if not mc_result.passed:
                logger.warning(
                    "[scheduler] Monte Carlo FAIL — p-value=%.4f >= 0.05. 실전 전환 불가.",
                    mc_result.p_value,
                )

        except Exception as exc:
            logger.error("[scheduler] Monte Carlo 검증 오류: %s", exc)
            if self._telegram:
                await self._telegram.send(
                    f"🚨 <b>Monte Carlo 검증 오류</b>\n{exc}",
                    priority=1,
                )

    def _load_oos_pnls(self, db_path: str) -> list[float]:
        """trades 테이블에서 실현 손익률 시퀀스 로드 (Monte Carlo 입력).

        Phase B 이후: Walk-Forward OOS 구간 거래만 필터링하도록 교체 예정.
        현재: 전체 SELL 체결 최근 500건 사용.

        Returns:
            pnl_pct 리스트 (빈 리스트 가능)
        """
        import sqlite3 as _sqlite3

        conn = _sqlite3.connect(db_path)
        try:
            rows = conn.execute(
                "SELECT pnl_pct FROM trades"
                " WHERE side='SELL' AND pnl_pct IS NOT NULL"
                " ORDER BY timestamp DESC LIMIT 500"
            ).fetchall()
            return [float(row[0]) for row in rows]
        finally:
            conn.close()

    def _save_backtest_results_sqlite(
        self,
        mc_result: MonteCarloResult,
        run_id: str,
        db_path: str,
    ) -> None:
        """Monte Carlo 결과를 backtest_results 테이블에 저장.

        Args:
            mc_result: MonteCarloResult 인스턴스
            run_id:    실행 식별자 (YYYYMMDD_HHMMSS)
            db_path:   SQLite 경로
        """
        import sqlite3 as _sqlite3

        conn = _sqlite3.connect(db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id          TEXT    NOT NULL,
                    run_at          TEXT    NOT NULL,
                    type            TEXT    NOT NULL,
                    actual_sharpe   REAL,
                    p_value         REAL,
                    edge_confidence REAL,
                    max_drawdown    REAL,
                    final_return    REAL,
                    n_trades        INTEGER,
                    passed          INTEGER NOT NULL,
                    summary_text    TEXT
                )
            """)
            conn.execute(
                """
                INSERT INTO backtest_results
                  (run_id, run_at, type,
                   actual_sharpe, p_value, edge_confidence,
                   max_drawdown, final_return, n_trades,
                   passed, summary_text)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    datetime.now(timezone.utc).isoformat(),
                    "monte_carlo",
                    round(mc_result.actual_sharpe,    4),
                    round(mc_result.p_value,          6),
                    round(mc_result.edge_confidence,  4),
                    round(mc_result.actual_max_drawdown, 4),
                    round(mc_result.actual_final_return, 4),
                    mc_result.n_trades,
                    int(mc_result.passed),
                    mc_result.summary(),
                ),
            )
            conn.commit()
            logger.info("[scheduler] backtest_results 저장 완료 run_id=%s", run_id)
        finally:
            conn.close()

    # ──────────────────────────────────────────────────────────────
    # DRY_RUN 콘솔 출력
    # ──────────────────────────────────────────────────────────────

    def _print_startup_banner(self) -> None:
        dry = self._cfg["dry_run"]
        mode_str = "🔵 DRY_RUN (페이퍼 트레이딩)" if dry else "🔴 LIVE (실거래)"
        phase = "Phase B" if self._cfg["phase_b"] else "Phase A"
        capital = self._cfg["initial_capital"]

        banner = f"""
╔══════════════════════════════════════════════════════╗
║         업비트 퀀트 자동매매 봇 v9 시작              ║
╠══════════════════════════════════════════════════════╣
║  모드     : {mode_str:<40}║
║  Phase    : {phase:<40}║
║  초기자본 : {capital:>12,.0f} 원                       ║
║  DB       : {str(self._cfg["db_path"]):<40}║
║  시각     : {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"):<40}║
╚══════════════════════════════════════════════════════╝
"""
        print(banner)
        if not dry:
            print("⚠️  경고: 실거래 모드입니다. trade_count < 200 이면 자동 DRY_RUN 전환됩니다.")

    # ──────────────────────────────────────────────────────────────
    # 비동기 루프들
    # ──────────────────────────────────────────────────────────────

    async def _main_schedule_loop(self) -> None:
        """5분 주기 메인 트레이딩 루프."""
        logger.info("[main] 메인 루프 시작 (주기=%ds)", MAIN_LOOP_INTERVAL_SEC)
        while not self._shutdown_event.is_set():
            try:
                # CandleBuilder에서 활성 코인 MarketState 수집
                market_states = await self._collect_market_states()
                if market_states:
                    await self._engine.main_loop(market_states)
                else:
                    logger.debug("[main] MarketState 없음 — WebSocket 데이터 수신 대기")
            except Exception as exc:
                logger.error("[main] 메인 루프 오류: %s", exc, exc_info=True)
                self._engine.circuit_breaker.record_api_error()

            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=MAIN_LOOP_INTERVAL_SEC,
                )
            except asyncio.TimeoutError:
                pass

    async def _position_monitor_loop(self) -> None:
        """1분 주기 포지션 모니터링 (트레일링 스탑 / 익절 / 손절)."""
        logger.info("[main] 포지션 모니터 루프 시작 (주기=%ds)", POSITION_LOOP_INTERVAL_SEC)
        while not self._shutdown_event.is_set():
            try:
                market_states = await self._collect_market_states()
                if market_states and hasattr(self._engine, "_position_loop"):
                    await self._engine._position_loop(market_states)
            except Exception as exc:
                logger.error("[main] 포지션 모니터 오류: %s", exc, exc_info=True)

            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=POSITION_LOOP_INTERVAL_SEC,
                )
            except asyncio.TimeoutError:
                pass

    async def _circuit_breaker_loop(self) -> None:
        """10초 주기 서킷브레이커 감시."""
        logger.info("[main] 서킷브레이커 루프 시작 (주기=%ds)", CIRCUIT_LOOP_INTERVAL_SEC)
        while not self._shutdown_event.is_set():
            try:
                self._engine.circuit_breaker.maybe_recover()
            except Exception as exc:
                logger.error("[main] 서킷브레이커 루프 오류: %s", exc)

            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=CIRCUIT_LOOP_INTERVAL_SEC,
                )
            except asyncio.TimeoutError:
                pass

    async def _telegram_loop(self) -> None:
        """텔레그램 봇 폴링 루프."""
        logger.info("[main] 텔레그램 봇 시작")
        try:
            await self._telegram.start(shutdown_event=self._shutdown_event)
        except Exception as exc:
            logger.error("[main] 텔레그램 봇 오류: %s", exc, exc_info=True)

    async def _storage_heartbeat_loop(self) -> None:
        """1시간 주기 디스크 점검."""
        while not self._shutdown_event.is_set():
            try:
                self._storage.check_disk_usage()
            except Exception as exc:
                logger.error("[main] 스토리지 점검 오류: %s", exc)

            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=DISK_CHECK_INTERVAL_SEC,
                )
            except asyncio.TimeoutError:
                pass

    # ──────────────────────────────────────────────────────────────
    # 헬퍼: MarketState 수집
    # ──────────────────────────────────────────────────────────────

    async def _collect_market_states(self):
        """활성 페어리스트의 MarketState 수집.

        get_market_state()가 async(run_in_executor 격리)이므로 이 메서드도 async.
        """
        from schema import MarketState
        states = []
        if self._collector is None:
            logger.warning("[main] collector 미초기화 — MarketState 수집 불가")
            return states
        try:
            pairs = self._collector.pairlist.get_active_pairs()
            for coin in pairs:
                ms = await self._collector.get_market_state(coin)
                if ms is not None:
                    states.append(ms)
        except Exception as exc:
            logger.warning("[main] MarketState 수집 오류: %s", exc)
        return states

    # ──────────────────────────────────────────────────────────────
    # graceful shutdown
    # ──────────────────────────────────────────────────────────────

    def _handle_signal(self, sig: signal.Signals) -> None:
        sig_name = sig.name
        logger.info("[main] 시그널 수신: %s — graceful shutdown 시작", sig_name)
        self._shutdown_event.set()

    async def _shutdown(self) -> None:
        """종료 시퀀스.

        순서: 텔레그램 알림 전송 → 잠시 대기 → 스케줄러 → 수집기 → 엔진 → DB
        """
        logger.info("[main] 종료 시퀀스 시작")

        # 1. 텔레그램 종료 알림 먼저 전송 후 대기
        if self._telegram:
            try:
                await self._telegram.send("🛑 봇 종료됨", priority=1)
                # 메시지가 실제로 전송될 시간을 확보
                await asyncio.sleep(SHUTDOWN_TELEGRAM_WAIT_SEC)
                await self._telegram.stop()
            except Exception:
                # 종료 중이므로 텔레그램 실패는 삼킴 — 로깅 시스템도 곧 닫힘
                pass

        # 2. APScheduler 종료
        if self._scheduler and self._scheduler.running:
            try:
                self._scheduler.shutdown(wait=False)
                logger.info("[main] APScheduler 종료")
            except Exception:
                # 이미 종료된 스케줄러이거나 이벤트 루프 상태 이상 — 무시 후 계속
                pass

        # 3. 데이터 수집기 종료
        if self._collector:
            try:
                await self._collector.stop()
            except Exception:
                # WebSocket/aiohttp 세션이 이미 닫혔을 수 있음 — 삼키고 계속
                pass

        # 4. 엔진 종료 (ProcessPoolExecutor 포함)
        if self._engine:
            try:
                self._engine.shutdown()
            except Exception:
                # ProcessPoolExecutor가 이미 종료됐을 수 있음 — 삼키고 계속
                pass

        # 5. DB 연결 종료
        if self._db_conn:
            try:
                self._db_conn.close()
            except Exception:
                # DB가 이미 닫혔거나 파일 핸들 이상 — 프로세스 종료 시 OS가 정리
                pass

        logger.info("[main] 모든 컴포넌트 종료 완료")

    # ──────────────────────────────────────────────────────────────
    # 진입점
    # ──────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """전체 봇 실행."""

        # ── 초기화 ────────────────────────────────────────────────
        logger.info("[main] === 1단계: 환경변수 검증 완료 ===")

        logger.info("[main] === 2단계: DB 초기화 ===")
        self._step2_init_db()

        logger.info("[main] === 3단계: StorageManager 초기화 ===")
        self._step3_init_storage()

        logger.info("[main] === 4단계: WebSocket 수집기 초기화 ===")
        self._step4_init_collector()

        logger.info("[main] === 5단계: 텔레그램 봇 초기화 ===")
        self._step5_init_telegram()

        logger.info("[main] === 6단계: 엔진 + 레이어 초기화 ===")
        self._step6_init_engine()

        logger.info("[main] === 7단계: 봇 컨텍스트 연결 ===")
        self._step7_set_bot_context()

        logger.info("[main] === 8단계: APScheduler 크론 작업 등록 ===")
        self._init_scheduler()

        # ── DRY_RUN 콘솔 출력 ────────────────────────────────────
        self._print_startup_banner()

        # ── 시그널 핸들러 등록 ────────────────────────────────────
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle_signal, sig)

        # ── 텔레그램 "봇 시작됨" 알림 ────────────────────────────
        dry_label = "DRY_RUN" if self._cfg["dry_run"] else "🔴 LIVE"
        phase_label = "Phase B" if self._cfg["phase_b"] else "Phase A"
        await self._telegram.send(
            f"🤖 <b>업비트 퀀트봇 시작</b>\n"
            f"모드: {dry_label} | {phase_label}\n"
            f"자본: {self._cfg['initial_capital']:,.0f}원\n"
            f"시각: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            priority=1,
        )

        # ── APScheduler 시작 ──────────────────────────────────────
        if self._scheduler:
            self._scheduler.start()
            logger.info("[main] APScheduler 시작")

        # ── 병렬 실행 ─────────────────────────────────────────────
        logger.info("[main] === 메인 트레이딩 루프 시작 ===")
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(
                    self._collector.start(),
                    name="collector",
                )
                tg.create_task(
                    self._telegram_loop(),
                    name="telegram",
                )
                tg.create_task(
                    self._main_schedule_loop(),
                    name="main_loop",
                )
                tg.create_task(
                    self._position_monitor_loop(),
                    name="position_monitor",
                )
                tg.create_task(
                    self._circuit_breaker_loop(),
                    name="circuit_breaker",
                )
                tg.create_task(
                    self._storage_heartbeat_loop(),
                    name="storage_heartbeat",
                )
        except* asyncio.CancelledError:
            pass
        except* Exception as eg:
            for exc in eg.exceptions:
                logger.error("[main] TaskGroup 오류: %s", exc, exc_info=True)
        finally:
            await self._shutdown()


# ─────────────────────────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    """동기 진입점 — asyncio.run() 래퍼."""
    cfg = _load_config()
    app = BotApplication(cfg)
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        logger.info("[main] KeyboardInterrupt — 종료")


if __name__ == "__main__":
    main()
