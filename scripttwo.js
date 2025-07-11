
(function () {
    const part = "contacts/detail";
    const url = window.location.href;
    if (!(url.includes(part) && !url.startsWith(part) && !url.endsWith(part))) return;
    const attachTagObserver = target => {
        const log = () => {
            const list = Array.from(target.querySelectorAll(".tag"))
                .map(el => el.textContent.trim());
            localStorage.setItem("tags", JSON.stringify(list));
            console.log(list);
        };
        log();
        new MutationObserver(log).observe(target, { childList: true });
    };
    const rootObserver = new MutationObserver((mx, obs) => {
        const tg = document.querySelector(".tag-group");
        if (tg) {
            obs.disconnect();
            attachTagObserver(tg);
        }
    });
    rootObserver.observe(document.body, { childList: true, subtree: true });
})();