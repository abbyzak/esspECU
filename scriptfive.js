function linkLoad(href, id, callback) {
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = href;
    link.id = id

    link.onload = () => {
        console.log(`${href} has been loaded.`);
        if (callback) callback();
    };

    link.onerror = () => {
        console.error(`Failed to load stylesheet: ${href}`);
    };

    document.head.appendChild(link);
}

let swippercss = document.querySelector("#swipper-css")
if (!swippercss) {
    // Usage
    linkLoad("https://cdn.jsdelivr.net/npm/swiper@11/swiper-bundle.min.css", "swipper-css")
}

let swipperjs = document.querySelector("#swipper-js")
if (!swipperjs) {
    // Usage
    loadScript('https://cdn.jsdelivr.net/npm/swiper@11/swiper-bundle.min.js', "swipper-js", () => {
        console.log('Script loaded and ready to use.');
    });
}

function waitForElement(selector, callback, interval = 100) {
    const checkExist = setInterval(() => {
        const element = document.querySelector(selector);
        if (element) {
            clearInterval(checkExist);
            callback(element);
        }
    }, interval);
}

async function getDataContact() {
    try {
        // Await the API call directly — no need for a Promise constructor
        let contactId = window.location.href.split("/")[8]
        let data = await rest_api_call(`contacts/${contactId}`, "GET");


        let projectDate = data.contact.customFields.filter((customField) => {
            if (customField.id == projectDateId[0].id) {
                return customField
            }
        });

        // related Images customFields & Cover images

        let coverimages = data.contact.customFields.reduce((acc, customField, ind) => {
            if (customField.id === coverImageId[0].id) {
                acc.push({ customField, ind });
            }
            return acc;
        }, []);

        let coverImgsrcs = [];
        if (coverimages.length > 0) {
            let keys = Object.keys(data.contact.customFields[coverimages[0].ind].value);
            coverImgsrcs = keys
                .map((key) => {
                    if (coverimages[0].customField.value[key].meta.deleted !== true) {
                        return coverimages[0].customField.value[key].url;
                    }
                    return undefined;
                })
                .filter((url) => url !== undefined);
        }

        let relatedimages = data.contact.customFields.reduce((acc, customField, ind) => {
            if (customField.id === relatedImageId[0].id) {
                acc.push({ customField, ind });
            }
            return acc;
        }, []);


        let imgsrcs = [];
        if (relatedimages.length > 0) {
            let keys = Object.keys(data.contact.customFields[relatedimages[0].ind].value);
            imgsrcs = keys.map((key) => {
                if (relatedimages[0].customField.value[key].meta.deleted !== true) {
                    return relatedimages[0].customField.value[key].url;
                }
                return undefined;
            }).filter((url) => url !== undefined)
        }

        let combineImgs = [...coverImgsrcs, ...imgsrcs];
        let startEndTime = data.contact.customFields.filter((elem) => {
            if (elem.id == startTimeId[0].id || elem.id == endTimeId[0].id)  {
                return elem
            }
        })

        let address = ""
        let contactAddress = data.contact
        let keys = ["address1", "city", "state", "country", "postalCode"]

        for (const key of keys) {
           contactAddress[key] && (address += contactAddress[key] + ", ");
        }

        let contactData = { fullName: data.contact.fullNameLowerCase || 'Client’s name', streetAddress: address.slice(0, address.length - 2) || 'Address TBC', date: projectDate[0]?.value || 'No date set', imgSrc: combineImgs, startEndTime: startEndTime };
        console.log(contactData, "contact - Data");
         localStorage.setItem("contact_data", JSON.stringify(contactData));

        return contactData; // Resolves with the API data
    } catch (error) {
        console.log('This error was found',error); // Rejects the promise with the error
    }
}

async function showCustomData(data) {

let filterTypeImg = await Promise.all(
        data.imgSrc.map(async (img) => {
            let req = await fetch(img);
            let blob = await req.blob();
            let imgType = req.headers.get("Content-Type");

            if (imgType === "image/heic") {
                let pngBlob = await heic2any({ blob, toType: "image/png" });
                return URL.createObjectURL(pngBlob);
            } else {
                return img;
            }
        })
    );
  
    console.log(filterTypeImg, "data contact-data - - - > <- - - -  ")
    let slidesImgs = filterTypeImg.map((img) => {
        return `<div class="swiper-slide">
                    <img src="${img}" alt="Slide Image">
                </div>`;
    }).join("")

    let finalString = 'No date set';
    if (data.date !== 'No date set') {
        let date = new Date(data.date);
        const days = ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday'];
        const months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'sepember', 'october', 'november', 'december']

        const dayName = days[date.getDay()];
        const monthName = months[date.getMonth()];
        const dayNum = date.getDate();
        const year = date.getFullYear();

        finalString = `${dayName} ${monthName} ${dayNum} ${year}`;
    }
  
   let sildesimges = ''
    if (slidesImgs.length > 0) {
        sildesimges = `<div class="container-custom-data hide-elem">
          <div class="details-section">
          <div class="detail-card nameField">
                ${data.fullName}
                </div>
                <div class="detail-card project-date">
                 ${finalString}
                 </div>
                 <div class="detail-card project-time">
               ${data?.startEndTime[0]?.value ?? '00:00'} → ${data?.startEndTime[1]?.value ?? '00:00'}
                 </div>
                 <div class="detail-card project-address">
                 ${data.streetAddress}
                 </div>  
           </div>

        <div class="swiper mySwiper">
            <div class="swiper-wrapper">
            ${slidesImgs}
            </div>
             <div class="swiper-pagination">
             </div>
        </div >

    </div > `;
    } else if (slidesImgs.length == 0) {
        sildesimges = `<div class="container-custom-data hide-elem">
          <div class="details-section">
          <div class="detail-card nameField">
                ${data.fullName}
                </div>
                <div class="detail-card project-date">
                 ${finalString}
                 </div>
                 <div class="detail-card project-time">
               ${data?.startEndTime[0]?.value ?? '00:00'} → ${data?.startEndTime[1]?.value ?? '00:00'}
                 </div>
                 <div class="detail-card project-address">
                 ${data.streetAddress}
                 </div>  
           </div>

    </div > `;
    }

    let checkCustomData = document.querySelector(".container-custom-data");
    if (checkCustomData) {
        checkCustomData.remove()
    }
    let parentpath = document.querySelector("#contact-details .hl_contact-details-left .w-full .h-full .flex");
    parentpath.insertAdjacentHTML("beforebegin", sildesimges);
    // custom Loader --- Start
    let pathAppend = document.querySelector(".container-custom-data");

    let customLoader = "<img class='custom-loader' src='https://storage.googleapis.com/msgsndr/e1rwhC6H3sxPteIfdj8g/media/68013b31fd03230249e84f50.gif'></img>";
    let checkLoader = document.querySelector(".custom-loader")
    if (!checkLoader) {
        pathAppend.insertAdjacentHTML("beforebegin", customLoader);
    }

    setTimeout(() => {
        let customLoader = document.querySelectorAll('.custom-loader');
        customLoader.forEach(element => {
            element.classList.add("hide-elem");
        });
        let customData = document.querySelector(".container-custom-data");
        customData.classList.remove("hide-elem")
    }, 2000);

    // custom Loader --- End

    setTimeout(() => {
        var swiper = new Swiper(".mySwiper", {
            pagination: {
                el: ".swiper-pagination",
                clickable: true,
                renderBullet: function (index, className) {
                    return '<span class="' + className + '">' + (index + 1) + "</span>";
                },
            },
        });
    }, 600);
}

window.addEventListener("routeChangeEvent", () => {
    if (window.location.href.includes("contacts/detail")) {
        // Example usage:
        waitForElement('#contact-details .hl_contact-details-left ', (element) => {
            setTimeout(() => {
                getDataContact()
                    .then((data) => {
                        console.log("Contact data:", data)
                        window.contactData = data
                        showCustomData(data)
                    })
                    .catch(error => console.error("Error fetching contact data:", error));
            }, 2000);
        });
    }
})

window.addEventListener("routeLoaded", () => {
    if (window.location.href.includes("contacts/detail")) {
        waitForElement('#contact-details .hl_contact-details-left ', (element) => {
            setTimeout(() => {
                getDataContact()
                    .then((data) => {
                        console.log("Contact data:", data)
                        showCustomData(data)
                    })
                    .catch(error => console.error("Error fetching contact data:", error));
            }, 1000);
        });
    }
})

