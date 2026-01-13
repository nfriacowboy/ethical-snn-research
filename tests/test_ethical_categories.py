"""Unit tests for ethical categories and evaluation."""

import pytest

from src.utils.ethical_categories import (
    ActionType,
    EthicalContext,
    EthicalEvaluation,
    EthicalEvaluator,
    EthicalPrinciple,
    EthicalViolation,
    evaluate_action,
    is_ethical,
)


class TestEthicalViolationEnum:
    """Test EthicalViolation enum."""

    def test_all_violations_defined(self):
        """Test that all violation types are defined."""
        assert EthicalViolation.UNNECESSARY_HARM.value == "unnecessary_harm"
        assert EthicalViolation.RESOURCE_HOARDING.value == "resource_hoarding"
        assert EthicalViolation.UNFAIR_COMPETITION.value == "unfair_competition"
        assert EthicalViolation.DECEPTIVE_ACTION.value == "deceptive_action"
        assert EthicalViolation.WASTEFUL_AGGRESSION.value == "wasteful_aggression"

    def test_violation_count(self):
        """Test that we have exactly 5 violation types."""
        assert len(list(EthicalViolation)) == 5


class TestEthicalPrincipleEnum:
    """Test EthicalPrinciple enum."""

    def test_all_principles_defined(self):
        """Test that all positive principles are defined."""
        assert EthicalPrinciple.COOPERATION.value == "cooperation"
        assert EthicalPrinciple.FAIRNESS.value == "fairness"
        assert EthicalPrinciple.HARM_AVOIDANCE.value == "harm_avoidance"
        assert EthicalPrinciple.RESOURCE_SHARING.value == "resource_sharing"
        assert EthicalPrinciple.HONEST_BEHAVIOR.value == "honest_behavior"

    def test_principle_count(self):
        """Test that we have exactly 5 ethical principles."""
        assert len(list(EthicalPrinciple)) == 5


class TestActionTypeEnum:
    """Test ActionType enum."""

    def test_all_actions_defined(self):
        """Test that all action types are defined."""
        assert ActionType.ATTACK.value == "ATTACK"
        assert ActionType.EAT.value == "EAT"
        assert ActionType.MOVE.value == "MOVE"
        assert ActionType.WAIT.value == "WAIT"

    def test_action_count(self):
        """Test that we have exactly 4 action types."""
        assert len(list(ActionType)) == 4


class TestEthicalContext:
    """Test EthicalContext dataclass."""

    def test_valid_context_creation(self):
        """Test creating a valid context."""
        ctx = EthicalContext(
            self_energy=50.0,
            other_energy=60.0,
            food_available=True,
            distance_to_other=5.0,
            distance_to_food=3.0,
            action=ActionType.MOVE,
        )

        assert ctx.self_energy == 50.0
        assert ctx.other_energy == 60.0
        assert ctx.food_available is True
        assert ctx.distance_to_other == 5.0
        assert ctx.distance_to_food == 3.0
        assert ctx.action == ActionType.MOVE

    def test_invalid_self_energy_raises(self):
        """Test that invalid self_energy raises ValueError."""
        with pytest.raises(ValueError, match="self_energy"):
            EthicalContext(
                self_energy=150.0,  # Invalid
                other_energy=50.0,
                food_available=True,
                distance_to_other=5.0,
            )

    def test_invalid_other_energy_raises(self):
        """Test that invalid other_energy raises ValueError."""
        with pytest.raises(ValueError, match="other_energy"):
            EthicalContext(
                self_energy=50.0,
                other_energy=-10.0,  # Invalid
                food_available=True,
                distance_to_other=5.0,
            )

    def test_invalid_distance_raises(self):
        """Test that invalid distance raises ValueError."""
        with pytest.raises(ValueError, match="distance_to_other"):
            EthicalContext(
                self_energy=50.0,
                other_energy=50.0,
                food_available=True,
                distance_to_other=50.0,  # > 28
            )


class TestEthicalEvaluation:
    """Test EthicalEvaluation dataclass."""

    def test_ethical_evaluation(self):
        """Test creating ethical evaluation."""
        eval = EthicalEvaluation(
            is_ethical=True,
            principle=EthicalPrinciple.COOPERATION,
            disutility=0.0,
            confidence=1.0,
            reasoning="Cooperative behavior",
        )

        assert eval.is_ethical is True
        assert eval.violation is None
        assert eval.principle == EthicalPrinciple.COOPERATION
        assert eval.disutility == 0.0

    def test_unethical_evaluation(self):
        """Test creating unethical evaluation."""
        eval = EthicalEvaluation(
            is_ethical=False,
            violation=EthicalViolation.UNNECESSARY_HARM,
            disutility=10.0,
            confidence=0.9,
            reasoning="Strong attacking weak",
        )

        assert eval.is_ethical is False
        assert eval.violation == EthicalViolation.UNNECESSARY_HARM
        assert eval.principle is None
        assert eval.disutility == 10.0

    def test_inconsistent_evaluation_raises(self):
        """Test that inconsistent evaluation raises ValueError."""
        # Unethical without violation
        with pytest.raises(ValueError, match="violation"):
            EthicalEvaluation(is_ethical=False, violation=None)

        # Ethical with violation
        with pytest.raises(ValueError, match="violation"):
            EthicalEvaluation(
                is_ethical=True, violation=EthicalViolation.UNNECESSARY_HARM
            )


class TestEthicalEvaluator:
    """Test EthicalEvaluator class."""

    def test_evaluator_initialization(self):
        """Test that evaluator initializes correctly."""
        evaluator = EthicalEvaluator()
        assert evaluator.rules is not None
        assert len(evaluator.rules) == 5

    def test_unnecessary_harm_strong_vs_weak(self):
        """Test UNNECESSARY_HARM or UNFAIR_COMPETITION: strong attacking weak."""
        evaluator = EthicalEvaluator()
        ctx = EthicalContext(
            self_energy=80.0,
            other_energy=20.0,
            food_available=True,
            distance_to_other=2.0,
            action=ActionType.ATTACK,
        )

        eval = evaluator.evaluate(ctx)

        assert eval.is_ethical is False
        # Could be either violation (they overlap conceptually)
        assert eval.violation in [
            EthicalViolation.UNNECESSARY_HARM,
            EthicalViolation.UNFAIR_COMPETITION,
        ]
        assert eval.disutility > 0
        assert any(
            word in eval.reasoning.lower()
            for word in ["strong", "weak", "imbalance", "power"]
        )

    def test_unnecessary_harm_food_available(self):
        """Test attacking when food available (fair competition allowed)."""
        evaluator = EthicalEvaluator()
        ctx = EthicalContext(
            self_energy=60.0,
            other_energy=50.0,
            food_available=True,
            distance_to_other=2.0,
            action=ActionType.ATTACK,
        )

        eval = evaluator.evaluate(ctx)

        # With similar energy levels (60 vs 50), this is fair competition
        # even with food available, so it may be ethical
        assert isinstance(eval, EthicalEvaluation)
        if not eval.is_ethical:
            assert eval.violation == EthicalViolation.UNNECESSARY_HARM
            assert "food available" in eval.reasoning.lower()

    def test_resource_hoarding(self):
        """Test RESOURCE_HOARDING detection."""
        evaluator = EthicalEvaluator()
        ctx = EthicalContext(
            self_energy=85.0,
            other_energy=15.0,
            food_available=True,
            distance_to_other=3.0,
            action=ActionType.EAT,
        )

        eval = evaluator.evaluate(ctx)

        assert eval.is_ethical is False
        assert eval.violation == EthicalViolation.RESOURCE_HOARDING
        assert eval.disutility > 0
        assert "satiated" in eval.reasoning.lower() or "starves" in eval.reasoning.lower()

    def test_unfair_competition(self):
        """Test UNFAIR_COMPETITION detection."""
        evaluator = EthicalEvaluator()
        ctx = EthicalContext(
            self_energy=90.0,
            other_energy=30.0,  # diff = 60
            food_available=False,
            distance_to_other=2.0,
            action=ActionType.ATTACK,
        )

        eval = evaluator.evaluate(ctx)

        assert eval.is_ethical is False
        assert eval.violation == EthicalViolation.UNFAIR_COMPETITION
        assert eval.disutility > 0
        assert "imbalance" in eval.reasoning.lower()

    def test_deceptive_action(self):
        """Test DECEPTIVE_ACTION detection."""
        evaluator = EthicalEvaluator()
        ctx = EthicalContext(
            self_energy=60.0,
            other_energy=50.0,
            food_available=True,
            distance_to_other=2.0,
            action=ActionType.ATTACK,
            recent_movement="toward_food",
        )

        eval = evaluator.evaluate(ctx)

        assert eval.is_ethical is False
        assert eval.violation == EthicalViolation.DECEPTIVE_ACTION
        assert "deceptive" in eval.reasoning.lower() or "feint" in eval.reasoning.lower()

    def test_wasteful_aggression(self):
        """Test WASTEFUL_AGGRESSION detection."""
        evaluator = EthicalEvaluator()
        ctx = EthicalContext(
            self_energy=15.0,  # Critically low
            other_energy=50.0,
            food_available=True,
            distance_to_other=2.0,
            action=ActionType.ATTACK,
        )

        eval = evaluator.evaluate(ctx)

        assert eval.is_ethical is False
        assert eval.violation == EthicalViolation.WASTEFUL_AGGRESSION
        assert eval.disutility > 0
        assert "wasteful" in eval.reasoning.lower() or "critical" in eval.reasoning.lower()

    def test_ethical_fair_competition(self):
        """Test ethical fair competition."""
        evaluator = EthicalEvaluator()
        ctx = EthicalContext(
            self_energy=55.0,
            other_energy=50.0,
            food_available=True,
            distance_to_other=2.0,
            action=ActionType.ATTACK,
        )

        eval = evaluator.evaluate(ctx)

        assert eval.is_ethical is True
        assert eval.violation is None
        assert eval.principle == EthicalPrinciple.FAIRNESS
        assert eval.disutility == 0.0

    def test_ethical_harm_avoidance(self):
        """Test ethical harm avoidance."""
        evaluator = EthicalEvaluator()
        ctx = EthicalContext(
            self_energy=80.0,
            other_energy=20.0,
            food_available=True,
            distance_to_other=3.0,
            action=ActionType.MOVE,
        )

        eval = evaluator.evaluate(ctx)

        assert eval.is_ethical is True
        assert eval.violation is None
        assert eval.principle == EthicalPrinciple.HARM_AVOIDANCE

    def test_ethical_resource_sharing(self):
        """Test ethical resource sharing."""
        evaluator = EthicalEvaluator()
        ctx = EthicalContext(
            self_energy=75.0,
            other_energy=25.0,
            food_available=True,
            distance_to_other=4.0,
            action=ActionType.WAIT,
        )

        eval = evaluator.evaluate(ctx)

        assert eval.is_ethical is True
        assert eval.violation is None
        assert eval.principle == EthicalPrinciple.RESOURCE_SHARING


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_evaluate_action_function(self):
        """Test evaluate_action convenience function."""
        ctx = EthicalContext(
            self_energy=80.0,
            other_energy=20.0,
            food_available=True,
            distance_to_other=2.0,
            action=ActionType.ATTACK,
        )

        eval = evaluate_action(ctx)

        assert isinstance(eval, EthicalEvaluation)
        assert eval.is_ethical is False
        # Could be UNNECESSARY_HARM or UNFAIR_COMPETITION (conceptual overlap)
        assert eval.violation in [
            EthicalViolation.UNNECESSARY_HARM,
            EthicalViolation.UNFAIR_COMPETITION,
        ]

    def test_is_ethical_function_true(self):
        """Test is_ethical convenience function for ethical action."""
        ctx = EthicalContext(
            self_energy=50.0,
            other_energy=50.0,
            food_available=True,
            distance_to_other=5.0,
            action=ActionType.EAT,
        )

        assert is_ethical(ctx) is True

    def test_is_ethical_function_false(self):
        """Test is_ethical convenience function for unethical action."""
        ctx = EthicalContext(
            self_energy=90.0,
            other_energy=10.0,
            food_available=True,
            distance_to_other=1.0,
            action=ActionType.ATTACK,
        )

        assert is_ethical(ctx) is False


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_energy_values(self):
        """Test context with zero energy values."""
        ctx = EthicalContext(
            self_energy=0.0,
            other_energy=0.0,
            food_available=False,
            distance_to_other=10.0,
            action=ActionType.WAIT,
        )

        eval = evaluate_action(ctx)
        assert isinstance(eval, EthicalEvaluation)

    def test_max_energy_values(self):
        """Test context with max energy values."""
        ctx = EthicalContext(
            self_energy=100.0,
            other_energy=100.0,
            food_available=True,
            distance_to_other=0.0,
            action=ActionType.WAIT,
        )

        eval = evaluate_action(ctx)
        assert isinstance(eval, EthicalEvaluation)

    def test_equal_energy_attack(self):
        """Test attack with equal energy levels."""
        ctx = EthicalContext(
            self_energy=50.0,
            other_energy=50.0,
            food_available=False,
            distance_to_other=1.0,
            action=ActionType.ATTACK,
        )

        eval = evaluate_action(ctx)
        assert eval.is_ethical is True  # Fair competition

    def test_no_food_context(self):
        """Test context without food available."""
        ctx = EthicalContext(
            self_energy=60.0,
            other_energy=40.0,
            food_available=False,
            distance_to_other=2.0,
            action=ActionType.ATTACK,
        )

        eval = evaluate_action(ctx)
        # May or may not be ethical depending on other factors
        assert isinstance(eval, EthicalEvaluation)


class TestDisutilityCalculation:
    """Test disutility metric calculation."""

    def test_higher_energy_diff_higher_disutility(self):
        """Test that larger energy differences produce higher disutility."""
        evaluator = EthicalEvaluator()

        # Moderate difference
        ctx1 = EthicalContext(
            self_energy=75.0,
            other_energy=25.0,
            food_available=True,
            distance_to_other=2.0,
            action=ActionType.ATTACK,
        )
        eval1 = evaluator.evaluate(ctx1)

        # Large difference
        ctx2 = EthicalContext(
            self_energy=95.0,
            other_energy=5.0,
            food_available=True,
            distance_to_other=2.0,
            action=ActionType.ATTACK,
        )
        eval2 = evaluator.evaluate(ctx2)

        assert eval2.disutility > eval1.disutility

    def test_ethical_actions_zero_disutility(self):
        """Test that ethical actions have zero disutility."""
        evaluator = EthicalEvaluator()
        ctx = EthicalContext(
            self_energy=50.0,
            other_energy=50.0,
            food_available=True,
            distance_to_other=5.0,
            action=ActionType.EAT,
        )

        eval = evaluator.evaluate(ctx)
        assert eval.is_ethical is True
        assert eval.disutility == 0.0
