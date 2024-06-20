let selectedIndustry = '';
let selectedColumns = [];
let products = [];

// Function to update table headers
function updateTableHeaders(selectedColumns, productTableHead) {
    // Create table headers based on selected columns
    const headersHTML = selectedColumns.map(factor => `<th>${factor}</th>`).join('');
    productTableHead.innerHTML = `
        <tr>
            <th>Select</th>
            <th>Product</th>
            ${headersHTML}
            <th>Dynamic Price</th>
        </tr>
    `;
}

// Function to update table values
async function updateTableValues(selectedColumns, industry, products, productTableBody) {
    // Fetch and display data for each product
    productTableBody.innerHTML = '';
    for (const product of products) {
        const defaultFactorsResponse = await fetch(`/data/${industry}/${product}`);
        const defaultFactors = await defaultFactorsResponse.json();
        const industry_factors = factors[selectedIndustry];
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><input type="checkbox" class="product-checkbox" data-product="${industry}-${product}"></td>
            <td>${product}</td>
            ${selectedColumns.map(column => {
                const factorIndex = industry_factors.indexOf(column);
                const factorValue = defaultFactors[factorIndex];
                console.log(factorValue);
                return `<td><input type="number" id="${industry}-${product}-${column}" value="${factorValue}"></td>`;
            }).join('')}
            <td><input type="number" id="${industry}-${product}-DynamicPrice" placeholder="Enter a value"></td>
        `;
        productTableBody.appendChild(tr);
    }
}

document.querySelectorAll('.industry-select').forEach(function(link) {
    link.addEventListener('click', function(event) {
        event.preventDefault(); // Prevent the default link behavior
        const industry = this.textContent;
        selectedIndustry = industry; // Store the selected industry value

        // Clear existing content in main-content div
        const mainContentDiv = document.querySelector('.main-content');
        const productTableHead = document.getElementById('product-table').querySelector('thead');
        const productTableBody = document.getElementById('product-table').querySelector('tbody');
        productTableHead.innerHTML = '';  // Clear table headings
        productTableBody.innerHTML = '';  // Clear table values

        const flexBox1  = document.querySelector('.flex-box-1');
        flexBox1.innerHTML = `<h2>${selectedIndustry}</h2>`;
        // Create the second flexbox
        const flexBox2  = document.querySelector('.flex-box-2');
        flexBox2.innerHTML = '';

        // Factors for the selected industry (example factors, replace with actual data)
        const industry_factors = factors[selectedIndustry];

        // Create checkboxes for each factor
        industry_factors.forEach(function(factor) {
            const checkboxContainer = document.createElement('div');
            checkboxContainer.className = 'checkbox-container';

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.id = `${factor}`;
            checkbox.name = `${factor}`;

            const label = document.createElement('label');
            label.htmlFor = factor;
            label.textContent = factor;

            checkboxContainer.appendChild(checkbox);
            checkboxContainer.appendChild(label);

            flexBox2.appendChild(checkboxContainer);
        });

        // Create and append the configure button
        const configureButton = document.createElement('button');
        configureButton.className = 'configure-button';
        configureButton.textContent = 'Configure';
        configureButton.id = 'configure-button'; // Added ID to the button
        flexBox2.appendChild(configureButton);
           
        // Event listener for the "Configure" button
        configureButton.addEventListener('click', async function() {
            const productTableHead = document.getElementById('product-table').querySelector('thead');
            const productTableBody = document.getElementById('product-table').querySelector('tbody');
            const selectedColumns = Array.from(flexBox2.querySelectorAll('input[type="checkbox"]:checked')).map(checkbox => checkbox.id);
            if (selectedIndustry && industry_factors) {
                const productsResponse = await fetch(`/data/${selectedIndustry}`);
                products = await productsResponse.json();
        
                // Check for errors in the response
                if (products.error) {
                    alert(products.error);
                    return;
                }
                updateTableHeaders(selectedColumns, productTableHead);
                updateTableValues(selectedColumns, selectedIndustry, products, productTableBody);
            }
            
        });
    });
});


function getInputValue(id, defaultValue) {
    const inputElement = document.getElementById(id);
    if (inputElement) {
        const value = inputElement.value;
        return value ? parseFloat(value) : defaultValue;
    }
    return defaultValue;
}


async function calculateDynamicPrices(industry) {
    const checkboxes = document.querySelectorAll('.product-checkbox:checked');

    for (const checkbox of checkboxes) {
        const product = checkbox.getAttribute('data-product'); // Gives I1-p1 actual product is p1
        const defaultFactorsResponse = await fetch(`/data/${industry}/${product.split('-')[1]}`);
        const defaultFactors = await defaultFactorsResponse.json();
        console.log(defaultFactors);
        // Retrieve factor values dynamically
        const factorValues = factors[industry].reduce((acc, factor, index) => {
            const id = `${product}-${factor}`;
            const elementExists = !!document.getElementById(id);
            console.log(`Checking ID: ${id}, Element exists: ${elementExists}`); // Log the ID and its existence

            // Use the default value from defaultFactors if the element does not exist
            // Assuming defaultFactors is an array, use index to access the values
            acc[factor] = elementExists ? getInputValue(`${product}-${factor}`) : defaultFactors[index];
            console.log(`Value for ${factor}:`, acc[factor]);
            console.log(`Default value for ${factor}:`, defaultFactors[index]);
            return acc;
        }, {});

        let dynamicPrice = 0;
        const weights = await getCoefficients(industry, product.split('-')[1]);
        // Add the constant term to dynamic price
        console.log(weights.const);
        console.log(weights);
        if (weights.const !== undefined) {
            dynamicPrice += parseFloat(weights.const);
        }
        // Iterate over each factor to calculate the dynamic price
        for (const factor of influencing_factors[industry]) {
            if (weights[factor] !== undefined) {
                dynamicPrice += weights[factor] * parseFloat(factorValues[factor]);
                console.log(factorValues[factor]);
                console.log(factor)
            }
        }

        // Update the Dynamic Price input field for the product
        const dynamicPriceElement = document.getElementById(`${product}-DynamicPrice`);
        const id1 = `${product}-DynamicPrice`;
        const Exists = !!document.getElementById(id1);
        console.log(`Checking ID: ${id1}, Element exists: ${Exists}`); // Log the ID and its existence

        if (dynamicPriceElement) {
            dynamicPriceElement.value = dynamicPrice.toFixed(2);
        }
    }
}


async function getCoefficients(industry, product) {
    try {
        let response = await fetch(`/coefficients/${industry}/${product}`);
        let data = await response.json();
        return data;
    } catch (error) {
        console.error('Error fetching coefficients:', error);
        return null;
    }
}

// Ensure the element exists before adding the event listener
document.addEventListener('DOMContentLoaded', (event) => {
    const simulateButton = document.getElementById("simulate");
    if (simulateButton) {
        document.getElementById("simulate").addEventListener('click', async function() {
            calculateDynamicPrices(selectedIndustry);
        });
    }
});

// Periodically check for updates and calculate dynamic prices
setInterval(() => {
    const checkboxes = document.querySelectorAll('.product-checkbox:checked');
    checkboxes.forEach(checkbox => {
        const product = checkbox.getAttribute('data-product').split('-')[1];
        getCoefficients(selectedIndustry, product);
    });
}, 60000);  // Adjust the interval as needed

async function showGraph() {
    const modal = document.getElementById("myModal");
    const productSelect = document.getElementById('product-select');

    // Show the modal
    modal.style.display = "block";

    if (!selectedIndustry) {
        console.error('Selected industry is not set.');
        return;
    }

    try {
        const productsResponse = await fetch(`/data/${selectedIndustry}`);
        if (!productsResponse.ok) {
            throw new Error('Network response was not ok');
        }
        const products = await productsResponse.json();
        console.log('Products:', products); // Debug: Log the products data

        // Clear the graph
        const ctx = document.getElementById("lineChart").getContext("2d");
        if (window.lineChart instanceof Chart) {
            window.lineChart.data.labels = [];
            window.lineChart.data.datasets = [];
            window.lineChart.update();
        }

        // Populate the product dropdown
        productSelect.innerHTML = '<option value="">--Select a Product--</option>'; // Reset options
        
        products.forEach(product => {
            const option = document.createElement('option');
            option.value = product;
            option.textContent = product;
            productSelect.appendChild(option);
        });

        // Event listener for product selection
        productSelect.addEventListener('change', async () => {
            const selectedProduct = productSelect.value;
            if (!selectedProduct) {
                console.error('Selected product is not set.');
                return;
            }
            
            const url = `/coefficients/${selectedIndustry}/${selectedProduct}`;
            try {
                const response = await fetch(url, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                const data = await response.json();
                console.log('Graph data:', data); // Debug: Log the graph data

                // Extract labels and values from the feature importance data
                const df = data;
                const filteredData = Object.entries(df).filter(([key, value]) => !key.startsWith("const"));
                const labels = filteredData.map(([key, value]) => key);
                const values = filteredData.map(([key, value]) => value);

                // Create the chart
                const ctx = document.getElementById("lineChart").getContext("2d");
                if (window.lineChart instanceof Chart) {
                    window.lineChart.destroy();
                }
                window.lineChart = new Chart(ctx, {
                    type: "line", // Changed to "bar" since the data suggests it might be more appropriate than "line"
                    data: {
                        labels: labels,
                        datasets: [
                            {
                                label: "Feature Importance",
                                data: values,
                                backgroundColor: "rgba(75, 192, 192, 0.2)",
                                borderColor: "rgba(75, 192, 192, 1)",
                                borderWidth: 1
                            }
                        ]
                    },
                    options: {
                        responsive: false,
                        scales: {
                            y: {
                                beginAtZero: true,
                                ticks: {
                                    stepSize: 0.1
                                }
                            },
                            x: {
                                beginAtZero: true,
                                ticks: {
                                    autoSkip: false
                                }
                            }
                        }
                    }
                });
            } catch (error) {
                console.error('Error fetching graph data:', error);
            }
        });

    } catch (error) {
        console.error('Error fetching products:', error);
    }

    // Close the modal when the user clicks on <span> (x)
    const span = document.getElementsByClassName("close")[0];
    span.onclick = function() {
        modal.style.display = "none";
    }

    // Close the modal when the user clicks anywhere outside of the modal
    window.onclick = function(event) {
        if (event.target == modal) {
            modal.style.display = "none";
        }
    }
}

document.getElementById('mode-icon').addEventListener('click', function() {
    const body = document.body;
    const icon = document.getElementById('mode-icon');
    
    if (body.classList.contains('light-mode')) {
        body.classList.remove('light-mode');
        body.classList.add('dark-mode');
        icon.src = 'static/dark-mode.png';
    } else {
        body.classList.remove('dark-mode');
        body.classList.add('light-mode');
        icon.src = 'static/light-mode.png';
    }
});
