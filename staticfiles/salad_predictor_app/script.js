document.addEventListener('DOMContentLoaded', function() {
    // Your existing JS code for form handling
    // Update fetch URL to use Django's path:
    fetch('/predict/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCookie('csrftoken'),
        },
        body: JSON.stringify({ ingredients: ingredients })
    })
    .then(response => response.json())
    .then(data => {
        // Display results
    });
});