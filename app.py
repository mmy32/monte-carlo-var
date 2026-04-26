from flask import Flask, jsonify, render_template
import numpy as np
import multiprocessing as mp
import time

import var_sequential
import var_parallel

app = Flask(__name__)


def compute_var_es(pnl, alpha):
    var = np.percentile(pnl, (1 - alpha) * 100)
    tail = pnl[pnl <= var]
    es = tail.mean() if len(tail) > 0 else var
    return var, es


@app.route("/api/var")
def get_var():

    # Sequential timing
    start = time.perf_counter()
    pnl_seq = var_sequential.run_var_simulation()
    seq_time = time.perf_counter() - start

    # Parallel timing
    start = time.perf_counter()
    pnl_par = var_parallel.run_var_simulation()
    par_time = time.perf_counter() - start

    var_95, es_95 = compute_var_es(pnl_par, 0.95)
    var_99, es_99 = compute_var_es(pnl_par, 0.99)

    return jsonify({
        "var_95": float(var_95),
        "var_99": float(var_99),
        "es_95": float(es_95),
        "es_99": float(es_99),

        "returns": pnl_par.tolist(),

        "seq_time": float(seq_time),
        "par_time": float(par_time)
    })


@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    app.run(debug=True)