function createPopup() {
  const overlay = document.createElement('div');
  overlay.id = 'popup-overlay';
  overlay.className = 'fixed inset-0 bg-black bg-opacity-20 hidden z-50 flex justify-center items-start pt-24';
  overlay.style.cssText = `
    background: rgba(0,0,0,0.2);
  `;

  const popup = document.createElement('div');
  popup.id = 'popup';
  popup.className = 'ny-gradient border border-gray-200 w-full max-w-lg mx-4 p-8 ny-shadow relative slide-up overflow-hidden';
  popup.style.cssText = `
  display: none;
  max-height: 80vh;
  min-height: 400px;
  width: 100%;
  max-width: 480px;
`;


  // Add subtle background pattern
  const bgPattern = document.createElement('div');
  bgPattern.className = 'absolute inset-0 opacity-5';
  bgPattern.style.cssText = `
    background-image: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23000000' fill-opacity='0.1'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
  `;
  popup.appendChild(bgPattern);

  // Close button with animation
  const closeButton = document.createElement('button');
  closeButton.className = 'absolute top-4 right-6 w-7 h-7 bg-white border border-gray-300 hover:border-gray-400 flex items-center justify-center transition-all duration-300 hover:scale-110 group z-10';
  closeButton.innerHTML = '<span class="text-black text-lg font-bold group-hover:rotate-90 transition-transform duration-300">×</span>';
  closeButton.addEventListener('click', () => {
    gsap.to(popup, { scale: 0.8, opacity: 0, duration: 0.3 });
    gsap.to(overlay, { opacity: 0, duration: 0.3, onComplete: () => {
      overlay.style.display = 'none';
      popup.style.display = 'none';
      currentStep = 0;
      resetPopup();
    }});
  });
  popup.appendChild(closeButton);

  // Progress indicator
  const progressContainer = document.createElement('div');
  progressContainer.className = 'flex justify-center items-center mb-8 relative z-10';
  
  const progressSteps = ['Verify', 'Options', 'Design'];
  progressSteps.forEach((stepName, index) => {
    const stepContainer = document.createElement('div');
    stepContainer.className = 'flex items-center';
    
    const stepCircle = document.createElement('div');
    stepCircle.className = `w-10 h-10 flex items-center justify-center text-sm font-semibold step-indicator transition-all duration-300 ${index === 0 ? 'bg-black text-white active' : 'bg-gray-200 text-gray-700 border border-gray-300'}`;
    stepCircle.textContent = index + 1;
    stepCircle.id = `step-${index}`;
    
    stepContainer.appendChild(stepCircle);
    
    if (index < progressSteps.length - 1) {
      const connector = document.createElement('div');
      connector.className = 'w-16 h-0.5 bg-gray-300 mx-2';
      connector.id = `connector-${index}`;
      stepContainer.appendChild(connector);
    }
    
    progressContainer.appendChild(stepContainer);
  });
  popup.appendChild(progressContainer);

  // Step 1: Verification List
  const verificationStep = document.createElement('div');
  verificationStep.id = 'verification-step';
  verificationStep.className = 'fade-in relative z-10';
  verificationStep.style.cssText = `
    display: none;
    opacity: 0;
    transition: opacity 0.3s ease;
  `;

  const title1 = document.createElement('h2');
  title1.className = 'text-2xl font-bold text-black mb-2';
  title1.textContent = 'Verify Recipients';
  verificationStep.appendChild(title1);

  const subtitle1 = document.createElement('p');
  subtitle1.className = 'text-gray-600 mb-6 text-sm';
  subtitle1.textContent = 'Review and manage your recipient list';
  verificationStep.appendChild(subtitle1);

  const recipientsList = document.createElement('div');
  recipientsList.id = 'recipients-list';
  recipientsList.className = 'mb-6 max-h-48 overflow-y-auto custom-scrollbar space-y-2';
  verificationStep.appendChild(recipientsList);

  const nextButton = document.createElement('button');
  nextButton.className = 'w-full bg-white hover:bg-gray-50 text-black font-semibold py-3 px-6 transition-all duration-300 hover:scale-105 hover:shadow-lg flex items-center justify-center group border border-gray-300 hover:border-gray-400';
  nextButton.innerHTML = 'Continue <span class="ml-2 group-hover:translate-x-1 transition-transform duration-300">→</span>';
  nextButton.addEventListener('click', () => {
    currentStep = 1;
    updateStep();
  });
  verificationStep.appendChild(nextButton);

  // Step 2: Checkboxes
  const checkboxStep = document.createElement('div');
  checkboxStep.id = 'checkbox-step';
  checkboxStep.className = 'fade-in relative z-10';
  checkboxStep.style.cssText = `
    display: none;
    opacity: 0;
    transition: opacity 0.3s ease;
  `;

  const title2 = document.createElement('h2');
  title2.className = 'text-2xl font-bold text-black mb-2';
  title2.textContent = 'Options';
  checkboxStep.appendChild(title2);

  const subtitle2 = document.createElement('p');
  subtitle2.className = 'text-gray-600 mb-6 text-sm';
  subtitle2.textContent = 'Configure your mailing preferences';
  checkboxStep.appendChild(subtitle2);

  const checkbox1Div = document.createElement('div');
  checkbox1Div.className = 'mb-6 p-4 glass-effect';
  
  const checkbox1Wrapper = document.createElement('div');
  checkbox1Wrapper.className = 'flex items-center mb-3';
  
  const checkbox1 = document.createElement('input');
  checkbox1.type = 'checkbox';
  checkbox1.checked = true;
  checkbox1.className = 'w-5 h-5 text-black bg-white border-gray-300 focus:ring-gray-500 focus:ring-2 mr-3';
  
  const label1 = document.createElement('label');
  label1.className = 'text-black font-medium';
  label1.textContent = 'Include Return Address';
  
  checkbox1Wrapper.appendChild(checkbox1);
  checkbox1Wrapper.appendChild(label1);
  checkbox1Div.appendChild(checkbox1Wrapper);
  
  const description = document.createElement('div');
  description.className = 'text-gray-600 text-sm leading-relaxed';
  description.textContent = 'Include the return address on the mailing when using this design. THIS ONLY applies to mailings sent via Standard mail. First Class mail requires a return address.';
  checkbox1Div.appendChild(description);
  checkboxStep.appendChild(checkbox1Div);

  const checkbox2Div = document.createElement('div');
  checkbox2Div.className = 'mb-6 p-4 glass-effect';
  
  const checkbox2Wrapper = document.createElement('div');
  checkbox2Wrapper.className = 'flex items-center';
  
  const checkbox2 = document.createElement('input');
  checkbox2.type = 'checkbox';
  checkbox2.className = 'w-5 h-5 text-black bg-white border-gray-300 focus:ring-gray-500 focus:ring-2 mr-3';
  
  const label2 = document.createElement('label');
  label2.className = 'text-black font-medium';
  label2.textContent = 'Override Default Return Address?';
  
  checkbox2Wrapper.appendChild(checkbox2);
  checkbox2Wrapper.appendChild(label2);
  checkbox2Div.appendChild(checkbox2Wrapper);
  checkboxStep.appendChild(checkbox2Div);

  const nextButton2 = document.createElement('button');
  nextButton2.className = 'w-full bg-white hover:bg-gray-50 text-black font-semibold py-3 px-6 transition-all duration-300 hover:scale-105 hover:shadow-lg flex items-center justify-center group border border-gray-300 hover:border-gray-400';
  nextButton2.innerHTML = 'Continue <span class="ml-2 group-hover:translate-x-1 transition-transform duration-300">→</span>';
  nextButton2.addEventListener('click', () => {
    currentStep = 2;
    updateStep();
  });
  checkboxStep.appendChild(nextButton2);

  // Step 3: Design Selection
  const designStep = document.createElement('div');
  designStep.id = 'design-step';
  designStep.className = 'fade-in relative z-10';
  designStep.style.cssText = `
    display: none;
    opacity: 0;
    transition: opacity 0.3s ease;
  `;

  const title3 = document.createElement('h2');
  title3.className = 'text-2xl font-bold text-black mb-2';
  title3.textContent = 'Select Design';
  designStep.appendChild(title3);

  const subtitle3 = document.createElement('p');
  subtitle3.className = 'text-gray-600 mb-6 text-sm';
  subtitle3.textContent = 'Choose your preferred design template';
  designStep.appendChild(subtitle3);

  const designDiv = document.createElement('div');
  designDiv.className = 'mb-6 p-4 glass-effect';
  
  const designLabel = document.createElement('label');
  designLabel.className = 'block text-black font-medium mb-3';
  designLabel.textContent = 'Design Template';
  designDiv.appendChild(designLabel);
  
  const selectWrapper = document.createElement('div');
  selectWrapper.className = 'relative';
  
  const select = document.createElement('select');
  select.className = 'w-full h-12 bg-white border border-gray-300 text-black px-4 focus:ring-2 focus:ring-black focus:border-transparent appearance-none cursor-pointer hover:border-gray-400 transition-all duration-300';
  
  const options = [
    { value: 'postcard', text: 'Postcard' },
    { value: 'gift-card', text: 'Gift Card' },
    { value: 'mail', text: 'Mail' }
  ];
  
  options.forEach(optionData => {
    const option = document.createElement('option');
    option.value = optionData.value;
    option.textContent = optionData.text;
    select.appendChild(option);
  });
  
  selectWrapper.appendChild(select);
  
  const selectIcon = document.createElement('div');
  selectIcon.className = 'absolute right-3 top-1/2 transform -translate-y-1/2 pointer-events-none';
  selectIcon.innerHTML = '<span class="text-black">▼</span>';
  selectWrapper.appendChild(selectIcon);
  
  designDiv.appendChild(selectWrapper);
  designStep.appendChild(designDiv);

  const buttonsDiv = document.createElement('div');
  buttonsDiv.className = 'flex justify-between items-center gap-4';
  
  const viewProof = document.createElement('button');
  viewProof.className = 'flex-1 bg-white hover:bg-gray-50 text-black font-medium py-3 px-6 transition-all duration-300 hover:scale-105 flex items-center justify-center group border border-gray-300 hover:border-gray-400';
  viewProof.innerHTML = '<span class="mr-2 group-hover:scale-110 transition-transform duration-300">👁</span>Show Proof';
  
  const submitButton = document.createElement('button');
  submitButton.className = 'flex-1 bg-white hover:bg-gray-50 text-black font-semibold py-3 px-6 transition-all duration-300 hover:scale-105 hover:shadow-lg flex items-center justify-center group relative overflow-hidden border border-gray-300 hover:border-gray-400';
  submitButton.innerHTML = '<span class="mr-2 group-hover:translate-x-1 transition-transform duration-300">✈</span>Submit';
  
  // Add pulse ring effect to submit button
  const pulseRing = document.createElement('div');
  pulseRing.className = 'absolute inset-0 bg-gray-300 opacity-20 pulse-ring';
  submitButton.appendChild(pulseRing);
  
  buttonsDiv.appendChild(viewProof);
  buttonsDiv.appendChild(submitButton);
  designStep.appendChild(buttonsDiv);

  popup.appendChild(verificationStep);
  popup.appendChild(checkboxStep);
  popup.appendChild(designStep);

  overlay.appendChild(popup);
  document.body.appendChild(overlay);

  let currentStep = 0;
  let selectedContacts = [];

  function updateStep() {
    const steps = [verificationStep, checkboxStep, designStep];
    
    // Update progress indicators
    for (let i = 0; i <= currentStep; i++) {
      const stepCircle = document.getElementById(`step-${i}`);
      const connector = document.getElementById(`connector-${i}`);
      
      stepCircle.className = stepCircle.className.replace('bg-gray-200 text-gray-700 border border-gray-300', 'bg-black text-white active');
      if (connector) {
        connector.className = connector.className.replace('bg-gray-300', 'bg-black');
      }
    }
    
    steps.forEach((step, index) => {
      if (index === currentStep) {
        step.style.display = 'block';
        gsap.fromTo(step, { opacity: 0, y: 20 }, { opacity: 1, y: 0, duration: 0.5 });
      } else {
        gsap.to(step, { opacity: 0, y: -20, duration: 0.3, onComplete: () => {
          step.style.display = 'none';
        }});
      }
    });
  }

  function updateRecipientsList() {
    const list = recipientsList;
    list.innerHTML = '';
    
    if (selectedContacts.length === 0) {
      const emptyState = document.createElement('div');
      emptyState.className = 'text-center py-8 text-gray-500';
      emptyState.innerHTML = '<i class="fas fa-users text-3xl mb-3 block"></i><p>No recipients selected</p>';
      list.appendChild(emptyState);
      return;
    }
    
    selectedContacts.forEach((contact, index) => {
      const item = document.createElement('div');
      item.className = 'flex items-center justify-between p-3 glass-effect hover:bg-black hover:bg-opacity-10 transition-all duration-300';
      
      const nameSpan = document.createElement('div');
      nameSpan.className = 'flex items-center';
      nameSpan.innerHTML = `<i class="fas fa-user-circle text-lg mr-3 text-gray-700"></i><span class="text-black font-medium">${contact.name}</span>`;
      
      const removeButton = document.createElement('button');
      removeButton.className = 'bg-red-500 hover:bg-red-600 text-white py-1 px-3 text-sm font-medium transition-all duration-300 hover:scale-105 flex items-center';
      removeButton.innerHTML = '<i class="fas fa-times mr-1"></i>Remove';
      removeButton.addEventListener('click', () => {
        gsap.to(item, { scale: 0.8, opacity: 0, duration: 0.3, onComplete: () => {
          selectedContacts.splice(index, 1);
          updateRecipientsList();
        }});
      });
      
      item.appendChild(nameSpan);
      item.appendChild(removeButton);
      list.appendChild(item);
    });
  }

  function resetPopup() {
    selectedContacts = [];
    const contacts = window._contactsCollector && window._contactsCollector.getAllCollected() || [];
    selectedContacts = [...contacts];
    updateRecipientsList();
    currentStep = 0;
    updateStep();
  }

  return { overlay, popup, resetPopup };
}

const sendCardMenuWatcher = new MutationObserver(() => {
  const currentUrl = window.location.href;
  const topNav = document.querySelector('div.flex.flex-row.justify-start.items-center.topmenu-nav') || document.querySelector('.topmenu-nav');
  const alreadyExists = document.querySelector('.send-card-wrapper');
  const isContactsPage = currentUrl.includes('contacts/detail') || true;

  if (!isContactsPage || !topNav || alreadyExists) return;

  const wrapper = document.createElement('button');
  wrapper.type = 'button';
  wrapper.className = 'send-card-wrapper topmenu-navitem bg-white hover:bg-gray-50 text-black px-4 py-2 font-medium transition-all duration-300 hover:scale-105 hover:shadow-lg flex items-center gap-2 border-0 border-white-300 hover:border-gray-400';
  wrapper.innerHTML = '<i class="fas fa-paper-plane"></i>Send Cards';

  const { overlay, popup, resetPopup } = createPopup();

  wrapper.addEventListener('click', () => {
    resetPopup();
    overlay.style.display = 'flex';
    popup.style.display = 'block';
    gsap.fromTo(popup, { scale: 0.8, opacity: 0 }, { scale: 1, opacity: 1, duration: 0.5 });
    gsap.fromTo(overlay, { opacity: 0 }, { opacity: 1, duration: 0.3 });
  });

  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) {
      gsap.to(popup, { scale: 0.8, opacity: 0, duration: 0.3 });
      gsap.to(overlay, { opacity: 0, duration: 0.3, onComplete: () => {
        overlay.style.display = 'none';
        popup.style.display = 'none';
        currentStep = 0;
        resetPopup();
      }});
    }
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      gsap.to(popup, { scale: 0.8, opacity: 0, duration: 0.3 });
      gsap.to(overlay, { opacity: 0, duration: 0.3, onComplete: () => {
        overlay.style.display = 'none';
        popup.style.display = 'none';
        currentStep = 0;
        resetPopup();
      }});
    }
  });

  topNav.appendChild(wrapper);
});

sendCardMenuWatcher.observe(document.body, { childList: true, subtree: true });
