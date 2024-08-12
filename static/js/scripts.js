document.addEventListener("DOMContentLoaded", function() {
    var table = document.getElementById('missingValuesTable');
    var tableHeight = 400; // Adjust this value to set the desired height
    table.parentElement.style.overflowY = 'auto';
    table.parentElement.style.maxHeight = tableHeight + 'px';
});
