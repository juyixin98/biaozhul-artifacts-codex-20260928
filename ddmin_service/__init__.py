"""Controlled delta debugging service.

Minimizes structured inputs that reproduce a target failure of a
controlled local test program. Modules:

- models:     input model, enums, job specification
- classifier: failure classification (target / other / pass / indeterminate)
- oracle:     execution kernel + cache binding (program digest, env, input)
- reducer:    ddmin over the record-removal operator, checkpointable
- store:      SQLite evidence store (jobs, query cache, diagnostic events)
- service:    job orchestration (create / run / resume / report)
- api:        FastAPI report and control interface
"""

__version__ = "0.1.0"
