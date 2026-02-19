"""API routes package — combines all domain sub-routers into one."""

from fastapi import APIRouter, Depends

from api.middleware import verify_api_key
from api.routes.comparison import router as comparison_router
from api.routes.evt import router as evt_router
from api.routes.fractal import router as fractal_router
from api.routes.guardian import router as guardian_router
from api.routes.hawkes import router as hawkes_router
from api.routes.lvar import router as lvar_router
from api.routes.msm import router as msm_router
from api.routes.news import router as news_router
from api.routes.portfolio import router as portfolio_router
from api.routes.regime import router as regime_router
from api.routes.rough import router as rough_router
from api.routes.svj import router as svj_router
from api.routes.oracle import router as oracle_router
from api.routes.streams import router as streams_router
from api.routes.social import router as social_router
from api.routes.macro import router as macro_router
from api.routes.portfolio_risk import router as portfolio_risk_router
from api.routes.calibration_tasks import router as calibration_tasks_router
from api.routes.onchain import router as onchain_router
from api.routes.tick import router as tick_router
from api.routes.hawkes_onchain import router as hawkes_onchain_router
from api.routes.token import router as token_router
from api.routes.vine_copula import router as vine_copula_router
from api.routes.ccxt import router as ccxt_router
from api.routes.portfolio_opt import router as portfolio_opt_router
from api.routes.models import router as models_router
from api.routes.walk_forward import router as walk_forward_router
from api.routes.dexscreener import router as dexscreener_router
from api.routes.narrator import router as narrator_router
from api.routes.dx import router as dx_router
from api.routes.strategies import router as strategies_router
from api.routes.execution import router as execution_router
from api.routes.agents import router as agents_router
from api.routes.system import router as system_router

router = APIRouter(dependencies=[Depends(verify_api_key)])

router.include_router(msm_router)
router.include_router(news_router)
router.include_router(regime_router)
router.include_router(comparison_router)
router.include_router(portfolio_router)
router.include_router(evt_router)
router.include_router(hawkes_router)
router.include_router(fractal_router)
router.include_router(rough_router)
router.include_router(svj_router)
router.include_router(guardian_router)
router.include_router(lvar_router)
router.include_router(oracle_router)
router.include_router(streams_router)
router.include_router(social_router)
router.include_router(macro_router)
router.include_router(portfolio_risk_router)
router.include_router(calibration_tasks_router)
router.include_router(onchain_router)
router.include_router(tick_router)
router.include_router(hawkes_onchain_router)
router.include_router(token_router)
router.include_router(vine_copula_router)
router.include_router(ccxt_router)
router.include_router(portfolio_opt_router)
router.include_router(models_router)
router.include_router(walk_forward_router)
router.include_router(dexscreener_router)
router.include_router(narrator_router)
router.include_router(dx_router)
router.include_router(execution_router)
router.include_router(strategies_router)
router.include_router(agents_router)
router.include_router(system_router)

