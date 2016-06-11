(function($){
  $(function(){

    $('.button-collapse').sideNav();
    $('.parallax').parallax();

    $("#photo-button").on("click", function() {
      $("#image-upload").click();
    });

    $("#image-upload").on("change", function() {
      $("#submit").click();
    });

  }); // end of document ready
})(jQuery); // end of jQuery name space