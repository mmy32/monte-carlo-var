async function loadData() {
    const res = await fetch("/api/var");
    const data = await res.json();

    // ---- Text metrics ----
    document.getElementById("var").innerText =
        `VaR 95: $${data.var_95.toFixed(2)}, VaR 99: $${data.var_99.toFixed(2)}`;

    document.getElementById("es").innerText =
        `ES 95: $${data.es_95.toFixed(2)}, ES 99: $${data.es_99.toFixed(2)}`;

    // ---- Histogram ----
    const ctx1 = document.getElementById("histChart");

    const bins = 50;
    const values = data.returns;

    new Chart(ctx1, {
        type: "bar",
        data: {
            labels: values,
            datasets: [{
                label: "PnL Distribution",
                data: values,
            }]
        }
    });

    // ---- Performance chart ----
    const ctx2 = document.getElementById("perfChart");

    new Chart(ctx2, {
        type: "bar",
        data: {
            labels: ["Sequential", "Parallel"],
            datasets: [{
                label: "Runtime (seconds)",
                data: [data.seq_time, data.par_time],
                backgroundColor: ["blue", "green"]
            }]
        }
    });
}

loadData();