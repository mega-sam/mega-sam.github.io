document.addEventListener('DOMContentLoaded', () => {
  var options = {
    slidesToScroll: 2,
    slidesToShow: 2,
    loop: true,
    infinite: false,
    pagination: false,
    autoplay: true,
    autoplaySpeed: 5000,
  };

  // Initialize all div with carousel class
  var carousels = bulmaCarousel.attach('.carousel', options);
});
