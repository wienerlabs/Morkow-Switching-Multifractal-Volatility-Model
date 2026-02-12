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

