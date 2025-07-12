#!/usr/bin/env python3
"""
Clinical Scenarios Generator for Clarity Digital Twin
Generates realistic clinical scenarios for bipolar risk detection demo.
"""

import argparse
import json
import numpy as np
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any

class ClinicalScenarioGenerator:
    """Generate clinical scenarios for bipolar risk detection demo."""
    
    def __init__(self, condition: str = "bipolar", severity: str = "moderate"):
        self.condition = condition
        self.severity = severity
        self.scenario_id = str(uuid.uuid4())
        
        # Risk phase definitions
        self.risk_phases = {
            "baseline": {
                "duration_days": 30,
                "sleep_disruption": 0.1,
                "activity_variation": 0.15,
                "mood_instability": 0.2,
                "risk_score": 0.15
            },
            "prodromal": {
                "duration_days": 14,
                "sleep_disruption": 0.4,
                "activity_variation": 0.35,
                "mood_instability": 0.5,
                "risk_score": 0.45
            },
            "acute_manic": {
                "duration_days": 7,
                "sleep_disruption": 0.8,
                "activity_variation": 0.7,
                "mood_instability": 0.85,
                "risk_score": 0.9
            },
            "recovery": {
                "duration_days": 21,
                "sleep_disruption": 0.25,
                "activity_variation": 0.2,
                "mood_instability": 0.3,
                "risk_score": 0.25
            }
        }
        
        # Severity modifiers
        self.severity_modifiers = {
            "mild": 0.7,
            "moderate": 1.0,
            "severe": 1.3
        }
        
        self.modifier = self.severity_modifiers.get(severity, 1.0)
    
    def generate_baseline_scenario(self) -> Dict[str, Any]:
        """Generate baseline (normal) clinical scenario."""
        phase = self.risk_phases["baseline"]
        days = phase["duration_days"]
        
        scenario_data = []
        base_date = datetime.now() - timedelta(days=days)
        
        for i in range(days):
            date = base_date + timedelta(days=i)
            
            # Stable patterns with minimal variation
            sleep_hours = np.random.normal(7.5, 0.5)
            sleep_quality = np.random.normal(8.0, 0.8)
            activity_level = np.random.normal(7.0, 0.6)
            mood_score = np.random.normal(7.5, 0.5)
            
            # Biomarkers
            hrv_score = np.random.normal(40, 5)  # Healthy HRV
            stress_level = np.random.normal(3.0, 0.8)
            
            entry = {
                "date": date.isoformat(),
                "phase": "baseline",
                "day_in_phase": i + 1,
                "sleep_hours": round(max(6, sleep_hours), 2),
                "sleep_quality": round(max(1, min(10, sleep_quality)), 1),
                "activity_level": round(max(1, min(10, activity_level)), 1),
                "mood_score": round(max(1, min(10, mood_score)), 1),
                "hrv_score": round(max(20, hrv_score), 1),
                "stress_level": round(max(1, min(10, stress_level)), 1),
                "risk_score": round(phase["risk_score"] * self.modifier, 3),
                "risk_level": "low",
                "clinical_notes": "Normal baseline patterns, no concerning indicators"
            }
            
            scenario_data.append(entry)
        
        return {
            "scenario_type": "baseline",
            "condition": self.condition,
            "severity": self.severity,
            "duration_days": days,
            "data": scenario_data
        }
    
    def generate_prodromal_scenario(self) -> Dict[str, Any]:
        """Generate prodromal (early warning) scenario."""
        phase = self.risk_phases["prodromal"]
        days = phase["duration_days"]
        
        scenario_data = []
        base_date = datetime.now() - timedelta(days=days)
        
        for i in range(days):
            date = base_date + timedelta(days=i)
            
            # Gradual deterioration over time
            progression = (i + 1) / days  # 0 to 1 over the phase
            
            # Sleep disruption increases over time
            sleep_disruption = phase["sleep_disruption"] * progression * self.modifier
            sleep_hours = np.random.normal(7.5 - sleep_disruption * 2, 0.8)
            sleep_quality = np.random.normal(8.0 - sleep_disruption * 4, 1.2)
            
            # Activity becomes more variable
            activity_variation = phase["activity_variation"] * progression * self.modifier
            activity_level = np.random.normal(7.0, 0.6 + activity_variation * 3)
            
            # Mood becomes more unstable
            mood_instability = phase["mood_instability"] * progression * self.modifier
            mood_score = np.random.normal(7.5, 0.5 + mood_instability * 2)
            
            # Biomarkers show subtle changes
            hrv_score = np.random.normal(40 - progression * 10, 8)
            stress_level = np.random.normal(3.0 + progression * 2, 1.0)
            
            # Risk score increases
            risk_score = phase["risk_score"] * progression * self.modifier
            
            entry = {
                "date": date.isoformat(),
                "phase": "prodromal",
                "day_in_phase": i + 1,
                "progression": round(progression, 2),
                "sleep_hours": round(max(4, sleep_hours), 2),
                "sleep_quality": round(max(1, min(10, sleep_quality)), 1),
                "activity_level": round(max(1, min(10, activity_level)), 1),
                "mood_score": round(max(1, min(10, mood_score)), 1),
                "hrv_score": round(max(15, hrv_score), 1),
                "stress_level": round(max(1, min(10, stress_level)), 1),
                "risk_score": round(risk_score, 3),
                "risk_level": "moderate" if risk_score > 0.3 else "low",
                "clinical_notes": f"Subtle changes emerging, day {i+1} of prodromal phase"
            }
            
            scenario_data.append(entry)
        
        return {
            "scenario_type": "prodromal",
            "condition": self.condition,
            "severity": self.severity,
            "duration_days": days,
            "data": scenario_data
        }
    
    def generate_acute_manic_scenario(self) -> Dict[str, Any]:
        """Generate acute manic episode scenario."""
        phase = self.risk_phases["acute_manic"]
        days = phase["duration_days"]
        
        scenario_data = []
        base_date = datetime.now() - timedelta(days=days)
        
        for i in range(days):
            date = base_date + timedelta(days=i)
            
            # Severe disruption with high variability
            sleep_disruption = phase["sleep_disruption"] * self.modifier
            sleep_hours = np.random.normal(7.5 - sleep_disruption * 3, 1.5)
            sleep_quality = np.random.normal(8.0 - sleep_disruption * 5, 2.0)
            
            # High activity with extreme variation
            activity_variation = phase["activity_variation"] * self.modifier
            activity_level = np.random.normal(7.0, 0.6 + activity_variation * 4)
            
            # Mood highly unstable
            mood_instability = phase["mood_instability"] * self.modifier
            mood_score = np.random.normal(7.5, 0.5 + mood_instability * 3)
            
            # Biomarkers show significant changes
            hrv_score = np.random.normal(40 - 20, 12)
            stress_level = np.random.normal(3.0 + 4, 1.5)
            
            # High risk score
            risk_score = phase["risk_score"] * self.modifier
            
            entry = {
                "date": date.isoformat(),
                "phase": "acute_manic",
                "day_in_phase": i + 1,
                "sleep_hours": round(max(2, sleep_hours), 2),
                "sleep_quality": round(max(1, min(10, sleep_quality)), 1),
                "activity_level": round(max(1, min(10, activity_level)), 1),
                "mood_score": round(max(1, min(10, mood_score)), 1),
                "hrv_score": round(max(10, hrv_score), 1),
                "stress_level": round(max(1, min(10, stress_level)), 1),
                "risk_score": round(risk_score, 3),
                "risk_level": "high",
                "clinical_notes": f"Acute manic episode, day {i+1} - immediate intervention needed"
            }
            
            scenario_data.append(entry)
        
        return {
            "scenario_type": "acute_manic",
            "condition": self.condition,
            "severity": self.severity,
            "duration_days": days,
            "data": scenario_data
        }
    
    def generate_recovery_scenario(self) -> Dict[str, Any]:
        """Generate recovery/stabilization scenario."""
        phase = self.risk_phases["recovery"]
        days = phase["duration_days"]
        
        scenario_data = []
        base_date = datetime.now() - timedelta(days=days)
        
        for i in range(days):
            date = base_date + timedelta(days=i)
            
            # Gradual improvement over time
            recovery_progress = (i + 1) / days  # 0 to 1 over the phase
            
            # Sleep gradually improves
            sleep_disruption = phase["sleep_disruption"] * (1 - recovery_progress) * self.modifier
            sleep_hours = np.random.normal(7.5 - sleep_disruption * 1.5, 0.9)
            sleep_quality = np.random.normal(8.0 - sleep_disruption * 3, 1.1)
            
            # Activity stabilizes
            activity_variation = phase["activity_variation"] * (1 - recovery_progress) * self.modifier
            activity_level = np.random.normal(7.0, 0.6 + activity_variation * 2)
            
            # Mood stabilizes
            mood_instability = phase["mood_instability"] * (1 - recovery_progress) * self.modifier
            mood_score = np.random.normal(7.5, 0.5 + mood_instability * 1.5)
            
            # Biomarkers improve
            hrv_score = np.random.normal(40 - (1 - recovery_progress) * 15, 6)
            stress_level = np.random.normal(3.0 + (1 - recovery_progress) * 2.5, 1.0)
            
            # Risk score decreases
            risk_score = phase["risk_score"] * (1 - recovery_progress * 0.7) * self.modifier
            
            entry = {
                "date": date.isoformat(),
                "phase": "recovery",
                "day_in_phase": i + 1,
                "recovery_progress": round(recovery_progress, 2),
                "sleep_hours": round(max(5, sleep_hours), 2),
                "sleep_quality": round(max(1, min(10, sleep_quality)), 1),
                "activity_level": round(max(1, min(10, activity_level)), 1),
                "mood_score": round(max(1, min(10, mood_score)), 1),
                "hrv_score": round(max(20, hrv_score), 1),
                "stress_level": round(max(1, min(10, stress_level)), 1),
                "risk_score": round(risk_score, 3),
                "risk_level": "moderate" if risk_score > 0.3 else "low",
                "clinical_notes": f"Recovery phase, day {i+1} - gradual stabilization"
            }
            
            scenario_data.append(entry)
        
        return {
            "scenario_type": "recovery",
            "condition": self.condition,
            "severity": self.severity,
            "duration_days": days,
            "data": scenario_data
        }
    
    def generate_complete_scenario(self) -> Dict[str, Any]:
        """Generate a complete clinical scenario with all phases."""
        baseline = self.generate_baseline_scenario()
        prodromal = self.generate_prodromal_scenario()
        acute = self.generate_acute_manic_scenario()
        recovery = self.generate_recovery_scenario()
        
        # Combine all phases
        complete_data = []
        complete_data.extend(baseline["data"])
        complete_data.extend(prodromal["data"])
        complete_data.extend(acute["data"])
        complete_data.extend(recovery["data"])
        
        # Add sequence numbers
        for i, entry in enumerate(complete_data):
            entry["sequence_day"] = i + 1
        
        total_days = sum([
            baseline["duration_days"],
            prodromal["duration_days"],
            acute["duration_days"],
            recovery["duration_days"]
        ])
        
        return {
            "scenario_id": self.scenario_id,
            "scenario_type": "complete_episode",
            "condition": self.condition,
            "severity": self.severity,
            "total_duration_days": total_days,
            "phases": {
                "baseline": baseline["duration_days"],
                "prodromal": prodromal["duration_days"],
                "acute_manic": acute["duration_days"],
                "recovery": recovery["duration_days"]
            },
            "data": complete_data,
            "metadata": {
                "generation_date": datetime.now().isoformat(),
                "clinical_validity": "For demonstration purposes only",
                "synthetic_data": True,
                "risk_thresholds": {
                    "low": "< 0.3",
                    "moderate": "0.3 - 0.6",
                    "high": "> 0.6"
                }
            }
        }
    
    def generate_all_scenarios(self, output_dir: str) -> None:
        """Generate all scenario types and save to directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating clinical scenarios for {self.condition} ({self.severity})")
        
        # Generate individual scenarios
        scenarios = {
            "baseline_scenario.json": self.generate_baseline_scenario(),
            "prodromal_scenario.json": self.generate_prodromal_scenario(),
            "acute_manic_scenario.json": self.generate_acute_manic_scenario(),
            "recovery_scenario.json": self.generate_recovery_scenario(),
            "complete_episode_scenario.json": self.generate_complete_scenario()
        }
        
        # Save all scenarios
        for filename, scenario in scenarios.items():
            filepath = output_path / filename
            with open(filepath, 'w') as f:
                json.dump(scenario, f, indent=2, default=str)
            
            data_points = len(scenario["data"])
            print(f"✅ Saved {filename} ({data_points} data points)")
        
        # Generate summary
        self._generate_scenarios_summary(output_path, scenarios)
        
        print(f"🎉 Clinical scenarios generation complete! Output: {output_path}")
    
    def _generate_scenarios_summary(self, output_path: Path, scenarios: Dict) -> None:
        """Generate summary of all scenarios."""
        
        summary = {
            "scenario_summary": {
                "condition": self.condition,
                "severity": self.severity,
                "total_scenarios": len(scenarios),
                "scenario_types": list(scenarios.keys()),
                "total_data_points": sum(len(s["data"]) for s in scenarios.values())
            },
            "risk_detection_targets": {
                "baseline": "Should detect low risk (< 0.3)",
                "prodromal": "Should detect gradual risk increase",
                "acute_manic": "Should detect high risk (> 0.6)",
                "recovery": "Should detect risk decrease",
                "complete_episode": "Should track full clinical trajectory"
            },
            "validation_criteria": {
                "temporal_consistency": "All dates in sequence",
                "clinical_realism": "Risk scores align with clinical phases",
                "pattern_validity": "Biomarkers correlate with risk levels",
                "detection_sensitivity": "Early warning in prodromal phase"
            }
        }
        
        with open(output_path / "scenarios_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("📊 Generated scenarios summary")

def main():
    """Main function to run the clinical scenarios generator."""
    parser = argparse.ArgumentParser(description="Generate clinical scenarios for bipolar risk detection")
    parser.add_argument("--condition", default="bipolar", 
                       choices=["bipolar", "depression", "anxiety"],
                       help="Clinical condition type")
    parser.add_argument("--severity", default="moderate",
                       choices=["mild", "moderate", "severe"],
                       help="Severity level of the condition")
    parser.add_argument("--output", default="demo_data/clinical",
                       help="Output directory for generated scenarios")
    
    args = parser.parse_args()
    
    # Create generator and run
    generator = ClinicalScenarioGenerator(condition=args.condition, severity=args.severity)
    generator.generate_all_scenarios(args.output)

if __name__ == "__main__":
    main() 