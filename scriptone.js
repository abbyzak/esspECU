
(async function () {
    const fields = [
        'project_date',
        'first_name',
        'last_name',
        'grooms_name',
        'email',
        'secondary_email_brideplanner',
        'phone',
        'contact_person',
        'inquiry_date',
        'start_time',
        'end_time',
        'morning_ready_by',
        'time_going_to_ceremony',
        'ceremony_start_time',
        'morning_location_type',
        'reception_appointment_title',
        '_reception_service_start_time',
        '_reception_service_end_time',
        'reception_venue_name',
        'reception_venue_address',
        'bridal_room',
        'reception_arrival',
        'room_reveal_time',
        'reception_start',
        'reception_hmu_ready_by',
        'reception_entrance_time',
        'old_wedding_date',
        'new_wedding_date',
        'date_of_birth',
        'source',
        'assistant_invoice_deadline',
        'wedding__6_months_out_date',
        'wedding__3_months_out_date',
        'quote_expiry_date',
        'inquiry_submitted_at',
        'previous_inquiry_submitted_at',
        'availability_confirmed_at',
        'wedding_invoice_link',
        'wed_confirmation_questionnaire_link',
        'artist_wedding_doc',
        'client_wedding_doc',
        'proposal_link'
    ];

    const remainingFields = new Set(fields);

    function handleField(field) {
        const selector = `input[name="contact.${field}"]`;
        const el = document.querySelector(selector);
        if (el) {
            if (el.value) {
                localStorage.removeItem(field)
                localStorage.setItem(field, el.value);
            } else {
                el.addEventListener('input', () => {
                    localStorage.removeItem(field)
                    localStorage.setItem(field, el.value);
                }, { once: true });
            }
            remainingFields.delete(field);
        }
    }

    // Initial check for all fields
    for (let field of fields) {
        handleField(field);
    }

    // Observe DOM changes if some fields are still missing
    if (remainingFields.size > 0) {
        const observer = new MutationObserver(() => {
            for (let field of Array.from(remainingFields)) {
                handleField(field);
            }
            if (remainingFields.size === 0) {
                observer.disconnect();
            }
        });
        observer.observe(document.body, { childList: true, subtree: true });
    }
})();