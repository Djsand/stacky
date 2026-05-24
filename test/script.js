const header = document.querySelector("[data-header]");
const toggle = document.querySelector(".nav-toggle");
const navLinks = document.querySelectorAll(".site-nav a");

const syncHeader = () => {
  header.classList.toggle("is-scrolled", window.scrollY > 12);
};

toggle?.addEventListener("click", () => {
  const isOpen = header.classList.toggle("nav-open");
  toggle.setAttribute("aria-expanded", String(isOpen));
});

navLinks.forEach((link) => {
  link.addEventListener("click", () => {
    header.classList.remove("nav-open");
    toggle?.setAttribute("aria-expanded", "false");
  });
});

window.addEventListener("scroll", syncHeader, { passive: true });
syncHeader();
