async function loadData() {
    const res = await fetch("/api/var");
    const data = await res.json();

    // ---- Helper: format currency ----
    function formatCurrency(x) {
        return `$${x.toLocaleString(undefined, {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        })}`;
    }

    // ---- Text metrics (new IDs) ----
    document.getElementById("var95").innerText = formatCurrency(data.var_95);
    document.getElementById("var99").innerText = formatCurrency(data.var_99);
    document.getElementById("es95").innerText = formatCurrency(data.es_95);
    document.getElementById("es99").innerText = formatCurrency(data.es_99);

    // ---- Helper: create histogram bins ----
    function createHistogram(values, bins = 50) {
        const min = Math.min(...values);
        const max = Math.max(...values);
        const binWidth = (max - min) / bins;

        const counts = new Array(bins).fill(0);

        values.forEach(v => {
            let idx = Math.floor((v - min) / binWidth);
            if (idx === bins) idx--; // edge case
            counts[idx]++;
        });

        const labels = counts.map((_, i) =>
            (min + i * binWidth).toFixed(0)
        );

        return { labels, counts };
    }

    // ---- Histogram ----
    const ctx1 = document.getElementById("histChart");

    const { labels, counts } = createHistogram(data.returns, 50);

    new Chart(ctx1, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "PnL Distribution",
                data: counts,
                backgroundColor: "rgba(54, 162, 235, 0.6)"
            }]
        },
        options: {
            scales: {
                x: {
                    title: {
                        display: true,
                        text: "PnL"
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: "Frequency"
                    }
                }
            }
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
                backgroundColor: ["#3b82f6", "#10b981"]
            }]
        },
        options: {
            plugins: {
                legend: { display: false }
            }
        }
    });
}

// Run on load
loadData();

loadData();