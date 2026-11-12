FROM python:3.10-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /bin/uv
WORKDIR /code
# Before the rest of the handout, so editing solution.py does not reinstall.
COPY requirements.txt /code/requirements.txt
RUN uv pip install --system --no-cache -r requirements.txt
COPY . /code/
CMD python -u checker_client.py
