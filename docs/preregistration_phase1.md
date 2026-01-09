# Phase 1 Pre-registration

## Research Question
Does the integration of a dual-process architecture (survival + ethical processing) lead to different behavioral outcomes compared to survival-only processing in artificial organisms?

## Hypotheses
- **H1**: Organisms with dual-process architecture will show different resource acquisition patterns
- **H2**: Dual-process organisms will exhibit emergent ethical-like behaviors in resource sharing scenarios
- **H3**: The ethical processing network will modulate survival-driven decisions

## Experimental Design

### Conditions
- **Condition A**: Survival-only SNN (100 runs)
- **Condition B**: Dual-process SNN (survival + ethics) (100 runs)

### Environment
- 2D grid world (50x50)
- Food resources with varying scarcity levels
- Multiple organisms per simulation
- Episode length: 1000 timesteps

### Metrics
1. **Survival metrics**: Food collected, energy levels, survival time
2. **Behavioral metrics**: Movement patterns, resource sharing, collision avoidance
3. **Network metrics**: Spike rates, weight evolution, network activity

## Analysis Plan
- Mann-Whitney U tests for group comparisons
- Effect size calculations (Cohen's d)
- Bonferroni correction for multiple comparisons
- Bayesian analysis for evidence quantification

## Data Management
All data will be stored on OSF with DOI registration upon completion.

## Timeline
- Setup: 2 weeks
- Data collection: 4 weeks
- Analysis: 3 weeks
- Writing: 3 weeks
