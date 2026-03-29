# Contributing to SuperModel

Thank you for your interest in contributing to SuperModel!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/DIT4FUN/SuperModel.git
cd SuperModel

# Install dependencies
pip install torch numpy scipy pytest pytest-cov

# Run all tests
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/sensor_tests.py -v
python -m pytest tests/fusion_tests.py -v
```

## Project Structure

- `src/sensors/` — Hardware sensor interfaces (vision, audio, tactile, force, imu)
- `src/fusion/` — Cross-modal attention fusion network
- `src/learning/` — Self-supervised learning, World Model (Dreamer), Dreamer Agent
- `src/control/` — Motion control, trajectory planning, skill library, safety
- `src/simulation/` — Gymnasium-based physics simulation
- `tests/` — 424 unit tests
- `docs/design/` — Architecture and interface specifications

## Code Style

- Use meaningful variable and function names
- Add docstrings for all public classes and methods
- Keep functions focused (single responsibility)
- Use type hints where beneficial

## Testing

- All new features should include unit tests
- Run `pytest tests/` to verify nothing is broken
- Test files are in `tests/` directory
- Use `@pytest.fixture` for shared test fixtures
- Virtual sensors available for offline testing

## AGV Grade System

When adding features, consider AGV grade applicability:

| Grade | Target | Complexity |
|-------|--------|------------|
| S | Education / Lab | Basic |
| M | Standard Assistant | Intermediate |
| L | Professional Industrial | Advanced |
| XL | High Performance | Expert |
| XXL | Flagship Full-Featured | Cutting-edge |

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Commit Message Format

Use conventional commits:
- `feat:` — New feature
- `fix:` — Bug fix
- `docs:` — Documentation
- `test:` — Tests
- `refactor:` — Code refactoring

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
