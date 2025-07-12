#!/usr/bin/env python3
"""Chat Context Generator for Clarity Digital Twin
Generates conversation seeds and context for HealthKit chat demo.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any
import uuid


class ChatContextGenerator:
    """Generate chat context and conversation seeds for demo."""

    def __init__(self, persona: str = "health_conscious") -> None:
        self.persona = persona
        self.context_id = str(uuid.uuid4())

        # Persona-specific characteristics
        self.personas = {
            "health_conscious": {
                "age_range": "25-45",
                "activity_level": "active",
                "health_goals": ["fitness", "sleep_optimization", "stress_management"],
                "interests": ["nutrition", "exercise", "wellness", "biohacking"],
                "communication_style": "curious and analytical"
            },
            "clinical_patient": {
                "age_range": "30-60",
                "activity_level": "moderate",
                "health_goals": ["mood_stability", "medication_adherence", "episode_prevention"],
                "interests": ["mental_health", "treatment_outcomes", "early_warning_signs"],
                "communication_style": "concerned and seeking guidance"
            },
            "fitness_enthusiast": {
                "age_range": "20-40",
                "activity_level": "very_active",
                "health_goals": ["performance_optimization", "recovery", "training_insights"],
                "interests": ["workouts", "heart_rate_zones", "recovery_metrics"],
                "communication_style": "performance-focused and data-driven"
            }
        }

        self.persona_data = self.personas.get(persona, self.personas["health_conscious"])

    def generate_conversation_starters(self) -> list[dict[str, Any]]:
        """Generate conversation starter questions."""
        if self.persona == "health_conscious":
            starters = [
                {
                    "category": "sleep_analysis",
                    "question": "How has my sleep quality changed over the past month?",
                    "expected_response_type": "trend_analysis",
                    "data_needed": ["sleep_hours", "sleep_quality", "sleep_efficiency"],
                    "complexity": "medium"
                },
                {
                    "category": "activity_patterns",
                    "question": "Show me patterns in my activity levels during stressful periods",
                    "expected_response_type": "correlation_analysis",
                    "data_needed": ["steps", "stress_level", "activity_minutes"],
                    "complexity": "high"
                },
                {
                    "category": "wellness_trends",
                    "question": "What's my average heart rate variability this week compared to last month?",
                    "expected_response_type": "comparative_analysis",
                    "data_needed": ["hrv_rmssd", "dates"],
                    "complexity": "medium"
                },
                {
                    "category": "holistic_health",
                    "question": "How do my sleep, activity, and mood correlate with each other?",
                    "expected_response_type": "multi_modal_analysis",
                    "data_needed": ["sleep_hours", "steps", "mood_score"],
                    "complexity": "high"
                },
                {
                    "category": "goal_tracking",
                    "question": "Am I meeting my weekly exercise goals based on my Apple Watch data?",
                    "expected_response_type": "goal_assessment",
                    "data_needed": ["workout_duration", "active_minutes", "calories_burned"],
                    "complexity": "low"
                }
            ]

        elif self.persona == "clinical_patient":
            starters = [
                {
                    "category": "risk_assessment",
                    "question": "Am I at risk for a mood episode based on my recent data?",
                    "expected_response_type": "risk_analysis",
                    "data_needed": ["sleep_hours", "mood_score", "activity_level", "stress_level"],
                    "complexity": "high"
                },
                {
                    "category": "early_warning",
                    "question": "Have there been any concerning changes in my sleep patterns lately?",
                    "expected_response_type": "anomaly_detection",
                    "data_needed": ["sleep_hours", "sleep_quality", "bedtime", "wake_time"],
                    "complexity": "medium"
                },
                {
                    "category": "medication_correlation",
                    "question": "How has my mood stability been since starting my new medication?",
                    "expected_response_type": "intervention_analysis",
                    "data_needed": ["mood_score", "medication_dates", "side_effects"],
                    "complexity": "high"
                },
                {
                    "category": "symptom_tracking",
                    "question": "Show me my stress levels over the past two weeks",
                    "expected_response_type": "trend_visualization",
                    "data_needed": ["stress_level", "anxiety_level", "dates"],
                    "complexity": "low"
                },
                {
                    "category": "recovery_progress",
                    "question": "How has my overall wellness improved since my last episode?",
                    "expected_response_type": "recovery_analysis",
                    "data_needed": ["all_metrics", "episode_dates"],
                    "complexity": "high"
                }
            ]

        else:  # fitness_enthusiast
            starters = [
                {
                    "category": "performance_metrics",
                    "question": "What's my heart rate recovery trend after workouts?",
                    "expected_response_type": "performance_analysis",
                    "data_needed": ["heart_rate", "workout_data", "recovery_metrics"],
                    "complexity": "medium"
                },
                {
                    "category": "training_optimization",
                    "question": "Which days of the week do I perform best in my workouts?",
                    "expected_response_type": "pattern_analysis",
                    "data_needed": ["workout_performance", "day_of_week", "energy_levels"],
                    "complexity": "medium"
                },
                {
                    "category": "recovery_tracking",
                    "question": "How does my sleep quality affect my next-day workout performance?",
                    "expected_response_type": "correlation_analysis",
                    "data_needed": ["sleep_quality", "workout_performance", "hrv"],
                    "complexity": "high"
                },
                {
                    "category": "zone_analysis",
                    "question": "How much time do I spend in each heart rate zone during cardio?",
                    "expected_response_type": "zone_breakdown",
                    "data_needed": ["heart_rate_zones", "workout_duration", "workout_type"],
                    "complexity": "medium"
                },
                {
                    "category": "progress_tracking",
                    "question": "Show me my fitness progress over the last 3 months",
                    "expected_response_type": "progress_analysis",
                    "data_needed": ["workout_frequency", "performance_metrics", "body_composition"],
                    "complexity": "high"
                }
            ]

        # Add metadata to each starter
        for starter in starters:
            starter.update({
                "persona": self.persona,
                "id": str(uuid.uuid4()),
                "created_date": datetime.now().isoformat(),
                "priority": "high" if starter["complexity"] == "high" else "medium"
            })

        return starters

    def generate_followup_questions(self) -> list[dict[str, Any]]:
        """Generate follow-up questions for conversations."""
        followups = [
            {
                "trigger": "sleep_trend_shown",
                "question": "What factors might be contributing to these sleep changes?",
                "context": "User has seen their sleep trend data",
                "expected_depth": "deeper_analysis"
            },
            {
                "trigger": "high_stress_detected",
                "question": "Can you suggest some strategies to manage my stress levels?",
                "context": "System detected elevated stress patterns",
                "expected_depth": "actionable_recommendations"
            },
            {
                "trigger": "workout_performance_shown",
                "question": "How can I optimize my training based on these patterns?",
                "context": "User reviewed workout performance data",
                "expected_depth": "personalized_coaching"
            },
            {
                "trigger": "risk_assessment_completed",
                "question": "What specific warning signs should I watch for?",
                "context": "System provided risk assessment",
                "expected_depth": "clinical_guidance"
            },
            {
                "trigger": "correlation_revealed",
                "question": "Tell me more about why these metrics are connected",
                "context": "System showed correlations between health metrics",
                "expected_depth": "scientific_explanation"
            }
        ]

        for followup in followups:
            followup.update({
                "persona": self.persona,
                "id": str(uuid.uuid4()),
                "created_date": datetime.now().isoformat(),
                "type": "followup"
            })

        return followups

    def generate_response_templates(self) -> list[dict[str, Any]]:
        """Generate response templates for different query types."""
        templates = [
            {
                "response_type": "trend_analysis",
                "template": "Based on your {data_type} data over the past {time_period}, I can see {trend_description}. Your average {metric} was {value}, which is {comparison} compared to your baseline.",
                "visualization": "line_chart",
                "insights": ["trend_direction", "statistical_significance", "contextual_factors"],
                "recommendations": True
            },
            {
                "response_type": "correlation_analysis",
                "template": "I found a {correlation_strength} correlation between your {metric1} and {metric2}. When your {metric1} is {condition}, your {metric2} tends to be {outcome}.",
                "visualization": "scatter_plot",
                "insights": ["correlation_coefficient", "clinical_significance", "causation_notes"],
                "recommendations": True
            },
            {
                "response_type": "risk_analysis",
                "template": "Based on your recent patterns, your risk level is {risk_level}. Key factors contributing to this assessment include {risk_factors}. I recommend {actions}.",
                "visualization": "risk_dashboard",
                "insights": ["risk_score", "contributing_factors", "confidence_level"],
                "recommendations": True,
                "clinical_validation": True
            },
            {
                "response_type": "anomaly_detection",
                "template": "I detected {anomaly_count} unusual patterns in your {data_type} data. These occurred on {dates} and were characterized by {anomaly_description}.",
                "visualization": "anomaly_timeline",
                "insights": ["anomaly_severity", "potential_causes", "frequency"],
                "recommendations": True
            },
            {
                "response_type": "comparative_analysis",
                "template": "Comparing {period1} to {period2}, your {metric} changed by {change_amount}. This represents a {change_significance} improvement/decline in your {health_aspect}.",
                "visualization": "comparison_chart",
                "insights": ["statistical_test", "effect_size", "practical_significance"],
                "recommendations": True
            }
        ]

        for template in templates:
            template.update({
                "persona": self.persona,
                "id": str(uuid.uuid4()),
                "created_date": datetime.now().isoformat(),
                "version": "1.0"
            })

        return templates

    def generate_demo_conversations(self) -> list[dict[str, Any]]:
        """Generate complete demo conversation flows."""
        conversations = [
            {
                "conversation_id": str(uuid.uuid4()),
                "title": "Sleep Quality Investigation",
                "scenario": "User notices feeling tired despite sleeping 8 hours",
                "flow": [
                    {
                        "turn": 1,
                        "user": "I've been sleeping 8 hours but still feel tired. What's going on?",
                        "system": "Let me analyze your sleep data. I can see you're getting adequate sleep duration, but your sleep efficiency has dropped to 78% this week compared to your usual 85%. You're also experiencing more awakenings."
                    },
                    {
                        "turn": 2,
                        "user": "What might be causing the sleep disruption?",
                        "system": "Looking at your patterns, I notice your bedtime has been inconsistent, varying by up to 2 hours nightly. Also, your stress levels have been elevated (average 6.2/10) compared to your baseline of 4.5/10."
                    },
                    {
                        "turn": 3,
                        "user": "How can I improve my sleep quality?",
                        "system": "Based on your data, I recommend: 1) Establishing a consistent bedtime routine, 2) Practicing stress-reduction techniques before bed, 3) Consider tracking your evening activities to identify specific triggers."
                    }
                ],
                "data_sources": ["sleep_analysis", "stress_tracking", "bedtime_patterns"],
                "complexity": "medium",
                "duration_minutes": 5
            },
            {
                "conversation_id": str(uuid.uuid4()),
                "title": "Bipolar Risk Assessment",
                "scenario": "User is concerned about mood episode risk",
                "flow": [
                    {
                        "turn": 1,
                        "user": "I'm worried I might be heading toward a manic episode. What does my data show?",
                        "system": "I understand your concern. Let me analyze your recent patterns. Your risk score is currently 0.35 (moderate), up from 0.15 last month. Key factors include decreased sleep duration (5.8 hours vs 7.5 hour baseline) and increased activity variability."
                    },
                    {
                        "turn": 2,
                        "user": "What specific warning signs should I watch for?",
                        "system": "Based on your historical patterns, key warning signs include: sleep dropping below 5 hours for 3+ consecutive nights, dramatic mood swings (>3 point daily variations), and sustained high activity levels. Your current pattern shows early warning signs but hasn't reached critical thresholds."
                    },
                    {
                        "turn": 3,
                        "user": "What should I do to prevent an episode?",
                        "system": "I recommend: 1) Prioritize sleep hygiene - aim for 7+ hours nightly, 2) Contact your healthcare provider about these patterns, 3) Consider stress management techniques. I'll continue monitoring and alert you if patterns worsen."
                    }
                ],
                "data_sources": ["bipolar_risk_model", "sleep_tracking", "mood_tracking", "activity_patterns"],
                "complexity": "high",
                "duration_minutes": 8,
                "clinical_significance": "high"
            },
            {
                "conversation_id": str(uuid.uuid4()),
                "title": "Workout Performance Analysis",
                "scenario": "User wants to optimize training based on data",
                "flow": [
                    {
                        "turn": 1,
                        "user": "Which days do I perform best in my workouts?",
                        "system": "Great question! Analyzing your workout data, you perform best on Tuesdays and Thursdays. Your average power output is 15% higher, and your heart rate recovery is 25% faster on these days."
                    },
                    {
                        "turn": 2,
                        "user": "Why do I perform better on those days?",
                        "system": "Several factors align on your best days: you consistently sleep 7.5+ hours the night before, your HRV is 20% above baseline, and your stress levels are lower (3.2/10 vs 4.8/10 average). Monday rest day also contributes to Tuesday performance."
                    },
                    {
                        "turn": 3,
                        "user": "How can I optimize my other training days?",
                        "system": "To improve other days: 1) Prioritize sleep before Wednesday/Friday workouts, 2) Consider active recovery on Sundays instead of complete rest, 3) Monitor stress levels and adjust training intensity accordingly."
                    }
                ],
                "data_sources": ["workout_performance", "sleep_tracking", "hrv_monitoring", "stress_tracking"],
                "complexity": "high",
                "duration_minutes": 6
            }
        ]

        for conversation in conversations:
            conversation.update({
                "persona": self.persona,
                "created_date": datetime.now().isoformat(),
                "demo_ready": True
            })

        return conversations

    def generate_all_context(self, output_dir: str) -> None:
        """Generate all chat context and save to directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Generating chat context for persona: {self.persona}")

        # Generate all context types
        context_data = {
            "conversation_starters.json": self.generate_conversation_starters(),
            "followup_questions.json": self.generate_followup_questions(),
            "response_templates.json": self.generate_response_templates(),
            "demo_conversations.json": self.generate_demo_conversations()
        }

        # Save all context files
        for filename, data in context_data.items():
            filepath = output_path / filename
            with open(filepath, 'w', encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            print(f"✅ Saved {filename} ({len(data)} items)")

        # Generate context metadata
        self._generate_context_metadata(output_path, context_data)

        print(f"🎉 Chat context generation complete! Output: {output_path}")

    def _generate_context_metadata(self, output_path: Path, context_data: dict) -> None:
        """Generate metadata for the chat context."""
        metadata = {
            "context_summary": {
                "persona": self.persona,
                "context_id": self.context_id,
                "generation_date": datetime.now().isoformat(),
                "persona_characteristics": self.persona_data,
                "total_items": sum(len(data) for data in context_data.values())
            },
            "content_breakdown": {
                "conversation_starters": len(context_data["conversation_starters.json"]),
                "followup_questions": len(context_data["followup_questions.json"]),
                "response_templates": len(context_data["response_templates.json"]),
                "demo_conversations": len(context_data["demo_conversations.json"])
            },
            "demo_scenarios": {
                "complexity_levels": ["low", "medium", "high"],
                "data_sources": ["sleep", "activity", "mood", "stress", "hrv", "workouts"],
                "interaction_types": ["trend_analysis", "correlation_analysis", "risk_assessment", "anomaly_detection"]
            },
            "quality_assurance": {
                "natural_language": "Conversational and persona-appropriate",
                "clinical_accuracy": "Medically sound recommendations",
                "data_integration": "Multi-modal health data sources",
                "user_experience": "Intuitive and engaging interactions"
            }
        }

        with open(output_path / "context_metadata.json", 'w', encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print("📊 Generated context metadata")


def main() -> None:
    """Main function to run the chat context generator."""
    parser = argparse.ArgumentParser(description="Generate chat context for HealthKit demo")
    parser.add_argument("--persona", default="health_conscious",
                       choices=["health_conscious", "clinical_patient", "fitness_enthusiast"],
                       help="User persona for context generation")
    parser.add_argument("--output", default="demo_data/conversations",
                       help="Output directory for generated context")

    args = parser.parse_args()

    # Create generator and run
    generator = ChatContextGenerator(persona=args.persona)
    generator.generate_all_context(args.output)


if __name__ == "__main__":
    main()
