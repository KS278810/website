document.addEventListener("DOMContentLoaded", () => {

  // 1. Mobile Menu Toggle
  const mobileBtn = document.querySelector('.mobile-menu-btn');
  const nav = document.querySelector('.nav');

  mobileBtn.addEventListener('click', () => {
    nav.classList.toggle('is-open');
  });

  // 2. Smooth Scrolling for Anchor Links
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      e.preventDefault();

      nav.classList.remove('is-open'); // Close menu on click

      const targetId = this.getAttribute('href');
      if (targetId === '#') return;

      const targetElement = document.querySelector(targetId);
      if (targetElement) {
        // Adjust for sticky header height
        const headerOffset = 80;
        const elementPosition = targetElement.getBoundingClientRect().top;
        const offsetPosition = elementPosition + window.scrollY - headerOffset;

        window.scrollTo({
          top: offsetPosition,
          behavior: "smooth"
        });
      }
    });
  });

  // 3. Header Scroll Effect & Back to Top Button
  const header = document.querySelector('.header');
  const backToTop = document.getElementById('backToTop');

  window.addEventListener('scroll', () => {
    if (header) {
      if (window.scrollY > 50) {
        header.classList.add('scrolled');
      } else {
        header.classList.remove('scrolled');
      }
    }

    if (backToTop) {
      if (window.scrollY > 500) {
        backToTop.classList.add('visible');
      } else {
        backToTop.classList.remove('visible');
      }
    }
  });

  if (backToTop) {
    backToTop.addEventListener('click', () => {
      window.scrollTo({
        top: 0,
        behavior: "smooth"
      });
    });
  }

  // 4. Intersection Observer for Fade-In Animations (staggered by DOM order)
  const observerOptions = {
    root: null,
    rootMargin: '0px',
    threshold: 0.15
  };

  const observer = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
        observer.unobserve(entry.target); // Only animate once
      }
    });
  }, observerOptions);

  document.querySelectorAll('.fade-in').forEach(element => {
    observer.observe(element);
  });

  // 5. Restrained interactive tilt on the hero photo frame (desktop only,
  // respects prefers-reduced-motion). Adds a quiet, editorial touch without
  // gimmicky decoration — the frame subtly follows the cursor, no more.
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const photoFrame = document.querySelector('.photo-frame');
  const photo = photoFrame ? photoFrame.querySelector('.profile-photo') : null;
  const canHover = window.matchMedia('(hover: hover) and (pointer: fine)').matches;

  if (photoFrame && photo && !prefersReducedMotion && canHover) {
    const maxTiltDeg = 5;

    photoFrame.addEventListener('mousemove', (e) => {
      const rect = photoFrame.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width - 0.5;
      const y = (e.clientY - rect.top) / rect.height - 0.5;
      const rotateY = (-3 + x * maxTiltDeg).toFixed(2);
      const rotateX = (1.2 - y * maxTiltDeg).toFixed(2);
      photo.style.transform = `rotateY(${rotateY}deg) rotateX(${rotateX}deg)`;
    });

    photoFrame.addEventListener('mouseleave', () => {
      photo.style.transform = '';
    });
  }
});
