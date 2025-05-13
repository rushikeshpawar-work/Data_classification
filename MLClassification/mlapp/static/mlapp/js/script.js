// Function to add a new row to the table
function addRow() {
  const tableBody = document.getElementById('table-body');
  const row = document.createElement('tr');

  // Automatically count the number of <th> elements in the header row
  const headerCells = document.querySelectorAll('#data-table thead th');
  const columnCount = headerCells.length;

  for (let i = 0; i < columnCount; i++) {
    const cell = document.createElement('td');
    cell.innerHTML = '<input type="text" name="col' + i + '" />';
    row.appendChild(cell);
  }

  tableBody.appendChild(row);
}


// Function to render the prediction data frame as a table
function renderDataFrameTable(data) {
  const container = document.getElementById("df-table-container");
  container.innerHTML = ""; // clear any old data

  if (!data || data.length === 0) {
    container.innerHTML = "<p>No prediction data found.</p>";
    return;
  }

  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const tbody = document.createElement("tbody");

  // Table Headers
  const headers = Object.keys(data[0]);
  const headerRow = document.createElement("tr");
  headers.forEach(key => {
    const th = document.createElement("th");
    th.textContent = key;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);

  // Table Rows
  data.forEach(row => {
    const tr = document.createElement("tr");
    headers.forEach(key => {
      const td = document.createElement("td");
      td.textContent = row[key];
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  table.appendChild(thead);
  table.appendChild(tbody);
  container.appendChild(table);
}

// Global Chart reference
let accuracyChart;

// Function to dynamically update the confidence chart
function updateConfidenceChart(confidence) {
  const ctx = document.getElementById('accuracyChart').getContext('2d');

  const centerText = {
    id: 'centerText',
    beforeDraw(chart) {
      const { width, height } = chart;
      const ctx = chart.ctx;
      ctx.restore();
      const fontSize = (height / 5).toFixed(2);
      ctx.font = fontSize + "px Arial";
      ctx.textBaseline = "middle";
      ctx.fillStyle = "#2c3e50";
      const text = confidence + "%";
      const textX = Math.round((width - ctx.measureText(text).width) / 2);
      const textY = height / 2;
      ctx.fillText(text, textX, textY);
      ctx.save();
    }
  };

  // Destroy previous chart if it exists
  if (accuracyChart) {
    accuracyChart.destroy();
  }

  accuracyChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Accuracy', 'Remaining'],
      datasets: [{
        data: [confidence, 100 - confidence],
        backgroundColor: ['#2ecc71', '#ecf0f1'],
        borderWidth: 1
      }]
    },
    options: {
      cutout: '70%',
      plugins: {
        legend: { display: false },
        tooltip: { enabled: false },
        title: {
          display: true,
          text: `Model Confidence`,
          font: { size: 16 }
        }
      }
    },
    plugins: [centerText]
  });
}

// Handle form submission
document.getElementById('prediction-form').addEventListener('submit', function (e) {
  e.preventDefault();

  const rows = document.querySelectorAll('#table-body tr');
  const headers = Array.from(document.querySelectorAll('#data-table thead th')).map(th => th.innerText.trim());
  const selectedModel = document.getElementById("ml-models").value;

  const data = [];

  rows.forEach(row => {
    const inputs = row.querySelectorAll('input');
    const rowData = {};
    inputs.forEach((input, index) => {
      const key = headers[index];
      rowData[key] = input.value;
    });
    data.push(rowData);
  });

  const payload = {
    model: selectedModel,
    records: data
  };

  fetch('/predict/', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(payload)
  })
    .then(response => response.json())
    .then(result => {
      if (result.status === 'success') {
        renderDataFrameTable(result.data);
        document.getElementById('prediction-results').innerHTML = `<strong>✅ Prediction complete (${result.data.length} rows)</strong>`;

        if (typeof result.confidence === 'number') {
          updateConfidenceChart(result.confidence);
        }
      } else {
        alert("Prediction failed: " + result.message);
      }
    })
    .catch(error => {
      alert("Error: " + error);
    });
});

