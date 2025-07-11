/*
const weddingSummaryWatcher = new MutationObserver(() => {
    const heading = document.querySelector('div.topmenu-navtitle[role="heading"][aria-level="1"]');
    const topNavBar = document.querySelector('div.topmenu-nav.flex.flex-row.items-center');
    const existing = document.querySelector(".wedding-summary-wrapper");
    const requiredUrlPart = "v2/location/D0dpEPTZ1yR0jML6goZW/contacts/detail";
    const currentUrl = window.location.href;
    const isUrlMatch = currentUrl.includes(requiredUrlPart) && !currentUrl.startsWith(requiredUrlPart) && !currentUrl.endsWith(requiredUrlPart);

    if (heading && heading.textContent.trim() === "Contacts" && isUrlMatch) {
        if (topNavBar && !existing) {
            const wrapper = document.createElement("div");
            wrapper.className = "group text-left mx-1 pb-2 md:pb-3 text-sm font-medium topmenu-navitem cursor-pointer relative px-2 wedding-summary-wrapper";
            wrapper.style.lineHeight = "1.6rem";

            const span = document.createElement("span");
            span.className = "flex items-center text-black";
            const label = document.createElement("span");
            label.textContent = "Wedding Summary";

            const dropdown = document.createElement("div");
            dropdown.className = "absolute top-full mt-2 left-0 bg-white border border-gray-300 rounded shadow z-50 hidden";
            dropdown.style.minWidth = "160px";

            const styleColor = "background-color:#e9eafb; color:#5e4b8b;";
            const hoverStyle = "this.style.backgroundColor='#eee4fd'";
            const outStyle = "this.style.backgroundColor='#f9f4ff'";

            function showPopup(url, w = "98vw", h = "95vh") {
                const overlay = document.createElement("div");
                overlay.style.position = "fixed";
                overlay.style.top = "0";
                overlay.style.left = "0";
                overlay.style.width = "100%";
                overlay.style.height = "100%";
                overlay.style.background = "rgba(0,0,0,0.5)";
                overlay.style.display = "flex";
                overlay.style.alignItems = "center";
                overlay.style.justifyContent = "center";
                overlay.style.zIndex = "9999";

                const popup = document.createElement("div");
                popup.style.position = "relative";
                popup.style.width = w;
                popup.style.height = h;
                popup.style.background = "#fff";
                popup.style.border = "1px solid #ccc";
                popup.style.borderRadius = "12px";
                popup.style.boxShadow = "0 4px 16px rgba(0,0,0,0.2)";

                const iframe = document.createElement("iframe");
                iframe.src = url;
                iframe.style.width = "100%";
                iframe.style.height = "100%";
                iframe.style.border = "none";
                iframe.style.borderRadius = "12px";
                iframe.style.zoom = "0.9";

                const closeBtn = document.createElement("button");
                closeBtn.textContent = "✕";
                closeBtn.style.position = "absolute";
                closeBtn.style.top = "8px";
                closeBtn.style.right = "12px";
                closeBtn.style.background = "transparent";
                closeBtn.style.border = "none";
                closeBtn.style.fontSize = "20px";
                closeBtn.style.cursor = "pointer";
                closeBtn.onclick = () => document.body.removeChild(overlay);

                popup.appendChild(closeBtn);
                popup.appendChild(iframe);
                overlay.appendChild(popup);
                document.body.appendChild(overlay);
            }

            const btn1 = document.createElement("button");
            btn1.textContent = "Artist Summary";
            btn1.className = "block w-full text-left px-4 py-2 text-sm border-b border-gray-200";
            btn1.setAttribute("style", styleColor);
            btn1.onmouseover = () => btn1.style.backgroundColor = "#eee4fd";
            btn1.onmouseout = () => btn1.style.backgroundColor = "#f9f4ff";
            btn1.onclick = () => {
                const c = window.contact || window.contactData || {};
                const brideName = c.fullName?.trim() || "";
                let date = c.date || "";
                try { date = date ? new Date(date).toISOString().split("T")[0] : ""; } catch (e) { date = ""; }
                const address = c.streetAddress?.trim() || "";
                const imageUrl = Array.isArray(c.imgSrc) && c.imgSrc[0] ? c.imgSrc[0] : "";
                const q = new URLSearchParams({ brideName, date, address, imageUrl });
                showPopup("https://app.garagematch.nl/?" + q.toString());
            };

            const btn2 = document.createElement("button");
            btn2.textContent = "Bride Prep Guide";
            btn2.className = "block w-full text-left px-4 py-2 text-sm";
            btn2.setAttribute("style", styleColor);
            btn2.onmouseover = () => btn2.style.backgroundColor = "#eee4fd";
            btn2.onmouseout = () => btn2.style.backgroundColor = "#f9f4ff";
            btn2.onclick = () => {
                const c = window.contact || window.contactData || {};
                const params = new URLSearchParams({
                    brideName: c.fullName?.trim() || "",
                    date: c.date || "",
                    coverImageUrl: Array.isArray(c.imgSrc) && c.imgSrc[0] ? c.imgSrc[0] : "",
                    arrivalTime: c.startEndTime?.[0]?.value || "",
                    location: c.location || "",
                    address: c.streetAddress?.trim() || "",
                    parkingNote: c.parkingNote || "",
                    assignedArtists: (c.assignedArtists || []).join(",") || "",
                    mobile: c.mobile || "",
                    timeline: (c.timeline || []).join(",") || "",
                    timelineNote: c.timelineNote || "",
                    serviceCount: c.serviceCount || "",
                    readyBy: c.readyBy || "",
                    ceremonyTime: c.ceremonyTime || "",
                    ceremonyStart: c.ceremonyStart || "",
                    hairAccessories: (c.hairAccessories || []).join(",") || "",
                    hiredItemsNote: c.hiredItemsNote || "",
                    time: c.time || "",
                    notes: c.notes || "",
                    receptionImageUrls: Array.isArray(c.receptionImageUrls) ? c.receptionImageUrls.join(",") : "",
                    morningImageUrls: Array.isArray(c.morningImageUrls) ? c.morningImageUrls.join(",") : "",
                    galleryImages: Array.isArray(c.galleryImages) ? c.galleryImages.join(",") : "",
                    headingFont: c.headingFont || "",
                    bodyFont: c.bodyFont || "",
                    headingSize: c.headingSize || "",
                    bodySize: c.bodySize || "",
                    headingColor: c.headingColor || "",
                    bodyColor: c.bodyColor || "",
                    pageColor: c.pageColor || ""
                });
                showPopup("https://app.garagematch.nl/?artistsummary?" + params.toString());
            };

            const btn3 = document.createElement("button");
            btn3.innerHTML = "⚙️ Settings";
            btn3.className = "block w-full text-left px-4 py-2 text-sm";
            btn3.setAttribute("style", styleColor);
            btn3.onmouseover = () => btn3.style.backgroundColor = "#eee4fd";
            btn3.onmouseout = () => btn3.style.backgroundColor = "#f9f4ff";
            btn3.onclick = () => {
                showPopup("https://app.garagematch.nl/settings", "500px", "450px");
            };

            dropdown.appendChild(btn1);
            dropdown.appendChild(btn2);
            dropdown.appendChild(btn3);

            span.appendChild(label);
            wrapper.appendChild(span);
            wrapper.appendChild(dropdown);
            topNavBar.appendChild(wrapper);

            span.addEventListener("click", e => {
                e.stopPropagation();
                dropdown.classList.toggle("hidden");
            });

            document.addEventListener("click", e => {
                if (!wrapper.contains(e.target)) dropdown.classList.add("hidden");
            });
        }
    } else {
        const old = document.querySelector(".wedding-summary-wrapper");
        if (old) old.remove();
    }
});
weddingSummaryWatcher.observe(document.body, { childList: true, subtree: true });
*/
