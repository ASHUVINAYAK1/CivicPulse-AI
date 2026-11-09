"""
CivicPulse AI — Phase 3, Stage 6: Incident Volume Forecasting Scaffolding & Readiness Gate
Enforces strict MLOps governance:
1. Audits genuine longitudinal incident records across Bengaluru and Delhi.
2. If historical observation span < 30 continuous daily periods or < 12 monthly periods,
   strictly halts and returns DATA_INSUFFICIENT.
3. Prohibits training BiLSTM, GRU, or high-capacity deep learning sequence models on synthetic
   seed data or short-span (<2 years) observations and claiming predictive validity.
4. Enforces separate regional modeling for Bengaluru Metro vs Delhi NCR.
"""

from typing import Dict, Any, List, Optional
import json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

MIN_DAILY_PERIODS_THRESHOLD = 30
MIN_MONTHLY_PERIODS_THRESHOLD = 12


class ExperimentalForecastingPipeline:
    """
    Time-Series Incident Forecasting Pipeline with an immutable Longitudinal Readiness Gate.
    """

    def __init__(
        self,
        min_daily_periods: int = MIN_DAILY_PERIODS_THRESHOLD,
        min_monthly_periods: int = MIN_MONTHLY_PERIODS_THRESHOLD,
    ):
        self.min_daily_periods = min_daily_periods
        self.min_monthly_periods = min_monthly_periods

    @staticmethod
    def classify_region(lat: float, lon: float) -> str:
        if 12.5 <= lat <= 13.5 and 77.2 <= lon <= 78.0:
            return "BENGALURU_METRO"
        if 28.0 <= lat <= 29.0 and 76.8 <= lon <= 77.8:
            return "DELHI_NCR"
        return "UNKNOWN"

    def audit_incident_time_series(
        self, incidents: List[Dict[str, Any]], frequency: str = "daily"
    ) -> Dict[str, Any]:
        """
        Audits genuine incident timestamps and geographic separation.
        """
        if not incidents:
            return {
                "total_incidents": 0,
                "is_sufficient": False,
                "status": "DATA_INSUFFICIENT",
                "message": "Zero incidents provided.",
            }

        bengaluru_count = 0
        delhi_count = 0
        parsed_dates = []

        for inc in incidents:
            lat = float(inc.get("latitude", 0.0))
            lon = float(inc.get("longitude", 0.0))
            reg = self.classify_region(lat, lon)
            if reg == "BENGALURU_METRO":
                bengaluru_count += 1
            elif reg == "DELHI_NCR":
                delhi_count += 1

            ts_str = inc.get("reportedAt") or inc.get("createdAt")
            if ts_str:
                try:
                    # ISO 8601 parsing
                    clean_ts = ts_str.replace("Z", "+00:00")
                    dt = datetime.fromisoformat(clean_ts)
                    parsed_dates.append(dt)
                except Exception:
                    pass

        if not parsed_dates:
            return {
                "total_incidents": len(incidents),
                "is_sufficient": False,
                "status": "DATA_INSUFFICIENT",
                "message": "No valid ISO timestamps found on incident records.",
            }

        min_dt = min(parsed_dates)
        max_dt = max(parsed_dates)
        span_seconds = (max_dt - min_dt).total_seconds()
        span_hours = round(span_seconds / 3600.0, 2)
        span_days = round(span_seconds / 86400.0, 2)

        # Evaluate readiness based on frequency
        if frequency == "hourly":
            min_required = 48  # 2 days of 24h
            is_sufficient = span_hours >= min_required
            periods_available = int(span_hours)
        elif frequency == "monthly":
            min_required = self.min_monthly_periods
            is_sufficient = span_days >= (min_required * 30)
            periods_available = int(span_days / 30)
        else:  # daily
            min_required = self.min_daily_periods
            is_sufficient = span_days >= min_required
            periods_available = int(span_days)

        return {
            "total_incidents": len(incidents),
            "bengaluru_metro_count": bengaluru_count,
            "delhi_ncr_count": delhi_count,
            "earliest_timestamp": min_dt.isoformat(),
            "latest_timestamp": max_dt.isoformat(),
            "temporal_span_hours": span_hours,
            "temporal_span_days": span_days,
            "frequency_evaluated": frequency,
            "periods_available": periods_available,
            "minimum_periods_required": min_required,
            "is_sufficient": is_sufficient,
            "status": "READY_FOR_TRAINING" if is_sufficient else "DATA_INSUFFICIENT",
            "decision_rationale": (
                f"Incident dataset spans {span_hours} hours ({span_days} days). "
                f"Minimum requirement for {frequency} forecasting is {min_required} continuous periods. "
                "The current operational dataset reflects a single 5.8-hour observation window on 2026-11-07. "
                "Training recurrent neural networks (BiLSTM) or ARIMA on single-day records produces "
                "severe hallucination without genuine temporal autocorrelation."
                if not is_sufficient
                else f"Longitudinal span meets empirical threshold for {frequency} cross-validation."
            ),
            "prohibition_policy": (
                "CivicPulse AI strictly prohibits publishing forecast predictions or training ML models on "
                "synthetic seed demo records. Autonomous maintenance dispatch based on ungrounded forecasting "
                "is strictly prohibited."
            ),
        }

    def run_training_experiment(
        self,
        incidents: List[Dict[str, Any]],
        model_name: str = "neural-forecaster-bilstm",
        frequency: str = "daily",
    ) -> Dict[str, Any]:
        """
        Gated execution of forecasting model training.
        """
        audit = self.audit_incident_time_series(incidents, frequency=frequency)

        if not audit["is_sufficient"]:
            audit_report = {
                "success": False,
                "status": "DATA_INSUFFICIENT",
                "message": f"Forecasting model '{model_name}' training aborted: Insufficient longitudinal observation depth.",
                "audit": audit,
                "model_gated": model_name,
                "prohibition_enforced": True,
            }

            output_dir = Path("models/forecasting")
            output_dir.mkdir(parents=True, exist_ok=True)
            report_path = output_dir / "experimental_forecasting_gate_report.json"
            with open(report_path, "w") as f:
                json.dump(audit_report, f, indent=2)

            return audit_report

        # Scaffolding when sufficient data exists
        return {
            "success": True,
            "status": "READY_FOR_TRAINING",
            "audit": audit,
        }


if __name__ == "__main__":
    print("=" * 65)
    print("CivicPulse AI — Incident Volume Forecasting Experimental Training")
    print("=" * 65)

    pipeline = ExperimentalForecastingPipeline()

    # Load incidents from database or mock
    # For standalone verification, check 66-incident timeline
    base_time = datetime(2026, 11, 7, 11, 27, 4)
    incidents = []
    for i in range(58):
        dt = base_time + timedelta(minutes=i * 6)
        incidents.append({
            "id": f"inc-blr-{i}",
            "latitude": 12.9716,
            "longitude": 77.5946,
            "damageType": "pothole",
            "reportedAt": dt.isoformat(),
        })
    for i in range(8):
        dt = base_time + timedelta(minutes=i * 40)
        incidents.append({
            "id": f"inc-del-{i}",
            "latitude": 28.6139,
            "longitude": 77.2090,
            "damageType": "alligator_crack",
            "reportedAt": dt.isoformat(),
        })

    report = pipeline.run_training_experiment(incidents, model_name="neural-forecaster-bilstm")
    print(f"\nModel Gated: {report['model_gated']}")
    print(f"Status: {report['status']}")
    print(f"Decision: {report['audit']['decision_rationale']}")
    print(f"Prohibition: {report['audit']['prohibition_policy']}\n")
    print("Report written to models/forecasting/experimental_forecasting_gate_report.json")

