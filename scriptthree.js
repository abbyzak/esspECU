

(async function () { // Changed to async IIFE
    const styleId = "ios-calendar-style";
    const calendarSectionClass = "details-section";

    function showIframeOverDetails(url) {
        const center = document.querySelector(".hl_contact-details-center");
        const right = document.querySelector(".hl_contact-details-right");

        if (!center || !right) return;

        const centerRect = center.getBoundingClientRect();
        const rightRect = right.getBoundingClientRect();

        const iframeWidth = rightRect.right - centerRect.left;
        const iframeHeight = Math.max(centerRect.height, rightRect.height);

        const iframeOverlay = document.createElement("div");
        iframeOverlay.style.position = "absolute";
        iframeOverlay.style.left = `${centerRect.left + window.scrollX}px`;
        iframeOverlay.style.top = `${centerRect.top + window.scrollY}px`;
        iframeOverlay.style.width = `${iframeWidth}px`;
        iframeOverlay.style.height = `${iframeHeight}px`;
        iframeOverlay.style.zIndex = "9999";
        iframeOverlay.style.boxShadow = "0 4px 16px rgba(0,0,0,0.2)";
        iframeOverlay.style.borderRadius = "8px";
        iframeOverlay.style.overflow = "hidden";
        iframeOverlay.style.background = "#fff";

        const iframe = document.createElement("iframe");
        iframe.src = url;
        iframe.style.width = "100%";
        iframe.style.height = "100%";
        iframe.style.border = "none";

        const closeBtn = document.createElement("button");
        closeBtn.textContent = "✕";
        closeBtn.style.position = "absolute";
        closeBtn.style.top = "8px";
        closeBtn.style.right = "12px";
        closeBtn.style.zIndex = "10";
        closeBtn.style.background = "transparent";
        closeBtn.style.border = "none";
        closeBtn.style.fontSize = "20px";
        closeBtn.style.cursor = "pointer";

        closeBtn.onclick = () => {
            iframeOverlay.remove();
            center.style.visibility = "";
            right.style.visibility = "";
        };

        iframeOverlay.appendChild(closeBtn);
        iframeOverlay.appendChild(iframe);

        document.body.appendChild(iframeOverlay);

        center.style.visibility = "hidden";
        right.style.visibility = "hidden";
    }


    async function applyIOSCalendarLayout() {
        if (!window.location.href.includes("D0dpEPTZ1yR0jML6goZW/contacts/detail")) return
        const section = document.querySelector("." + calendarSectionClass);
        if (!section) return

        // Clear any existing calendar content
        const existingCalendar = section.querySelector(".calendar-header");
        if (existingCalendar) {
            section.innerHTML = "";
        }

        if (!document.getElementById(styleId)) {
            const s = document.createElement("style");
            s.id = styleId;
            s.textContent = `
        .${calendarSectionClass}{
          width:100%;
          background:#ffffff;
          border-radius:12px;
          font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
          box-shadow:0 1px 3px rgba(0,0,0,0.1);
          overflow:hidden;
        }
        .calendar-header{
          display:flex;
          flex-direction:column;
          align-items:flex-start;
          padding:9.6px 8px;
          background:#ffffff;
          border-bottom:0.5px solid #d1d1d6;
        }
        .calendar-title{
          font-size:19.6px;
          font-weight:600;
          color:#000000;
        }
        .calendar-date{
          font-size:12.4px;
          margin-left:10px;
          color:#D72626;
          font-weight:600;
        }
        .calendar-settings{
          width:20px;
          height:20px;
          display:flex;
          align-items:center;
          justify-content:center;
          cursor:pointer;
          position:absolute;
          top:9.6px;
          right:16px;
        }
        .calendar-settings:before{
          content:"⚙";
          font-size:14.8px;
          color:#8e8e93;
        }
        .calendar-body{
          padding:0;
          background:#ffffff;
        }
        
        /* Wedding Event Block */
        .detail-card.wedding{
          background:#D8E8FF;
          margin-bottom:2px;
          border-bottom:0.5px solid #d1d1d6;
        }
        .detail-card.wedding .card-indicator{
          background:#2E77D0;
        }
        .detail-card.wedding .card-title{
          color:#1B4F91;
        }
        .detail-card.wedding .card-subtitle{
          color:#4A4A4A;
        }
        .detail-card.wedding .card-time-start{
          color:#1B4F91;
        }
        .detail-card.wedding .card-time-end{
          color:#A3C0E5;
        }
        
        /* Reception Event Block */
        .detail-card.reception{
          background:#F1E8FB;
          margin:0;
          border-bottom:0.5px solid #d1d1d6;
        }
        .detail-card.reception .card-indicator{
          background:#9B5CBD;
        }
        .detail-card.reception .card-title{
          color:#7B3FA8;
        }
        .detail-card.reception .card-subtitle{
          color:#5F5F5F;
        }
        .detail-card.reception .card-time-start{
          color:#7B3FA8;
        }
        .detail-card.reception .card-time-end{
          color:#C69FE8;
        }
        
        /* Assistant Block */
        .detail-card.assistant{
          background:#FFF7E2;
          margin:0;
          border-bottom:0.5px solid #d1d1d6;
        }
        .detail-card.assistant .card-indicator{
          background:#D4AC0D;
        }
        .detail-card.assistant .card-title{
          color:#5A5A5A;
        }
        .detail-card.assistant .card-time-start{
          color:#7A5C44;
        }
        .detail-card.assistant .card-time-end{
          color:#A38B72;
        }
        .detail-card.assistant .card-icon{
          color:#5A5A5A;
        }
        
        .detail-card{
          display:flex;
          margin:0;
        }
        .detail-card:last-child{
          border-bottom:none;
        }
        .card-indicator{
          width:4px;
          flex-shrink:0;
        }
        .card-content{
          display:flex;
          justify-content:start;
          align-items:center;
          flex:1;
          padding:4px 12px;
          min-height:28px;
        }
        .card-text{
          flex:1;
          min-width:0;
          display:flex;
          flex-direction:column;
          justify-content:center;
        }
        .card-title{
          font-size:10.8px;
          font-weight:400;
          margin:0 0 1.6px 0;
          line-height:1.2;
        }
        .card-subtitle {
          font-size:9.2px;
          margin:0;
          line-height:1.2;
        }
        .card-time{
          font-size:9.2px;
          white-space:nowrap;
          margin-left:12px;
          text-align:right;
          line-height:1.2;
          display:flex;
          flex-direction:column;
        }
        .card-time-start{
          font-size:10.2px;
          font-weight:500;
        }
        .card-time-end{
          font-size:9.2px;
          font-weight:400;
        }
        .card-icon{
          width:25px;
          height:17px;
          margin-right:2px;
          margin-bottom:2px;
          display:flex-start;;
          align-items:center;
          justify-content:center;
          font-size:12.8px;
        }
      `;
            document.head.appendChild(s);
        }

        async function waitForLocalStorageKey(key, interval = 100, timeout = 10000) {
            return new Promise((resolve, reject) => {
                const start = Date.now();

                const check = () => {
                    const value = localStorage.getItem(key);
                    if (value !== null) {
                        resolve(value);
                    } else if (Date.now() - start > timeout) {
                        reject(new Error(`Timeout waiting for localStorage key: ${key}`));
                    } else {
                        setTimeout(check, interval);
                    }
                };

                check();
            });
        }

        // Extract real contact data from DOM elements
        const getContactData = async () => {
            function formatTimeString(timeStr) {
                const match = timeStr.match(/^(\d{1,2})(?::(\d{2}))?(am|pm)$/i);
                if (!match) return timeStr;

                let [, hour, minute, period] = match;
                hour = parseInt(hour, 10);
                minute = minute ? minute.padStart(2, '0') : '00';
                period = period.toUpperCase();

                return `${hour}:${minute} ${period}`;
            }
            function toTitleCase(str) {
                return str.split(' ').map(word =>
                    word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
                ).join(' ');
            }
            const database = JSON.parse(await waitForLocalStorageKey("contact_data"));
            return {
                name: toTitleCase(database?.fullName),
                date: database?.date,
                has_wedding_tag: true,
                morning_location_type: await waitForLocalStorageKey("morning_location_type"),
                address: contactData?.streetAddress,
                wedding_start: formatTimeString(database['startEndTime'][0].value),
                wedding_end: formatTimeString(database['startEndTime'][1].value),
                reception_start_time: formatTimeString(await waitForLocalStorageKey("reception_start_time")),
                reception_finish_time: formatTimeString(await waitForLocalStorageKey("reception_finish_time")),
                reception_appointment_title: await waitForLocalStorageKey("reception_appointment_title"),
                reception_venue_name: await waitForLocalStorageKey("reception_venue_name"),
                reception_venue_address: await waitForLocalStorageKey("reception_venue_address"),
                assistant_status: 'assistant confirmed',
                assistant_to_contact: 'Laura'
            };
        };

        const contact = await getContactData(); // Moved inside async IIFE and fixed syntax

        const entries = [];
        const tags = JSON.parse(localStorage.getItem("tags") || '[]');
        function hasTag(name) {
            return tags.includes(name);
        }
        // Block 1: Wedding Event (shows if contact has tag wedding booked)
        if (hasTag("wedding booked")) {
            entries.push({
                type: "wedding",
                title: contact.name + "'s Wedding",
                address: contact.morning_location_type + " - " + contact.address,
                start_time: contact.wedding_start,
                end_time: contact.wedding_end
            });
        }

        // Block 2: Reception Appointment (shows if all 3 custom fields are populated)
        if (contact.reception_start_time && contact.reception_finish_time && contact.reception_appointment_title) {
            entries.push({
                type: "reception",
                title: contact.reception_appointment_title,
                address: `${contact.reception_venue_name} - ${contact.reception_venue_address}`,
                start_time: contact.reception_start_time,
                end_time: contact.reception_finish_time
            });
        }

        // Block 3: Assistant (only shows if assistant_status = "assistant confirmed")
        if (contact.assistant_status === "assistant confirmed") {
            entries.push({
                type: "assistant",
                title: `Assistant ${contact.assistant_to_contact}`,
                address: "",
                start_time: "8:00 AM",
                end_time: "12:00 PM",
                icon: "👤"
            });
        }

        section.innerHTML = "";

        const header = document.createElement("div");
        header.className = "calendar-header";
        const openBtn = document.createElement("button");
        openBtn.textContent = "Open iFrame View";
        openBtn.style.cssText = `
        margin-top: 8px;
        padding: 6px 12px;
        font-size: 14px;
        background-color: #007bff;
        color: #fff;
        border: none;
        border-radius: 6px;
        cursor: pointer;
      `;
        openBtn.onclick = () => showIframeOverDetails("https://app.garagematch.nl/settings");

        header.appendChild(openBtn);


        const title = document.createElement("h1");
        title.className = "calendar-title";
        title.textContent = contact.name;

        const date = document.createElement("div");
        date.className = "calendar-date";
        date.textContent = contact.date;

        const settings = document.createElement("div");
        settings.className = "calendar-settings";

        header.append(title, date, settings);

        const body = document.createElement("div");
        body.className = "calendar-body";

        entries.forEach(entry => {
            const card = document.createElement("div");
            card.className = `detail-card ${entry.type}`;

            const indicator = document.createElement("div");
            indicator.className = "card-indicator";

            const content = document.createElement("div");
            content.className = "card-content";

            const text = document.createElement("div");
            text.className = "card-text";

            const titleEl = document.createElement("div");
            titleEl.className = "card-title";
            titleEl.textContent = entry.title;
            text.appendChild(titleEl);

            if (entry.address) {
                const sub = document.createElement("div");
                sub.className = "card-subtitle";
                sub.textContent = entry.address;
                text.appendChild(sub);
            }

            const timeEl = document.createElement("div");
            timeEl.className = "card-time";

            const startTime = document.createElement("div");
            startTime.className = "card-time-start";
            startTime.textContent = entry.start_time;

            const endTime = document.createElement("div");
            endTime.className = "card-time-end";
            endTime.textContent = entry.end_time;

            timeEl.appendChild(startTime);
            timeEl.appendChild(endTime);

            if (entry.icon) {
                const iconEl = document.createElement("div");
                iconEl.className = "card-icon";
                iconEl.textContent = entry.icon;
                content.append(iconEl, text, timeEl);
            } else {
                content.append(text, timeEl);
            }

            card.append(indicator, content);
            body.appendChild(card);
        });

        section.append(header, body);
    }

    let retryCount = 0;
    const maxRetries = 10;

    async function tryApplyLayout() {
        try {
            await applyIOSCalendarLayout();
            retryCount = 0;
        } catch (e) {
            console.error(`Layout application failed (attempt ${retryCount + 1}):`, e);
            retryCount++;

            if (retryCount < maxRetries) {
                const retryObserver = new MutationObserver(() => {
                    retryObserver.disconnect();
                    tryApplyLayout().catch(console.error);
                });

                retryObserver.observe(document.body, {
                    childList: true,
                    subtree: true,
                    attributes: true,
                    attributeFilter: ['value', 'class']
                });
            } else {
                console.error('Max retries reached, giving up');
                retryCount = 0;
            }
        }
    }

    const o = new MutationObserver(muts => {
        if (
            muts.some(m =>
                Array.from(m.addedNodes).some(
                    n =>
                        n.nodeType === 1 &&
                        (n.classList.contains(calendarSectionClass) ||
                            n.querySelector("." + calendarSectionClass))
                )
            )
        ) {
            setTimeout(() => tryApplyLayout().catch(console.error), 100);
        }
    });
    o.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ["class", "style"]
    });

    window.addEventListener("popstate", () =>
        setTimeout(() => tryApplyLayout().catch(console.error), 200)
    );

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", () =>
            setTimeout(() => tryApplyLayout().catch(console.error), 100)
        );
    }

    tryApplyLayout().catch(console.error);
})();
