$(document).ready(function(){main()});

function lungsOpacity(value){
  $(".result1_image").css("opacity", value)
}
function defeatsOpacity(value){
  $(".result2_image").css("opacity", value)
}

function make_result_info(data){
  text = "Affected <b>" + data.data.affections_square + "%</b> of lungs<br><br>"+
            "Left lung: " + data.data.left_affections + " affections<br>"+
            "Right lung: " + data.data.right_affections + " affections<br>"
  $(".info_container .res").html(text)
  $(".info_container .stats").html("Total: " + data.data.stats.all_time + " s.")
}

function main(){
    $("#files").change(function() {
        filename = this.files[0].name
        console.log(filename);
      });
    $(".refresh_page").click(function () {location.reload()})
    function sendData(form) {
      const XHR = new XMLHttpRequest();
      const FD = new FormData( form );
      XHR.addEventListener( "load", function(event) {
        $(".upload_container").removeClass("loading");
        res = event.target.responseText;
        res = $.parseJSON(res)
        console.log(res)
        $(".result_image").attr("src", res.data.img_url[0]);
        $(".result1_image").attr("src", res.data.img_url[1]);
        $(".result2_image").attr("src", res.data.img_url[2]);
        $(".upload_container").addClass("upload_container_right")
        $(".result_image, .result1_image, .result2_image").show(300)
        $(".info_container").show(300);
        make_result_info(res)
      } );
      XHR.addEventListener("error", function( event ) {
        alert("Error!");
        location.reload();
      } );
      XHR.open("POST", "//"+document.domain+":5000/api/predict");
      XHR.send(FD);
    }
    $("form").change(function(event) 
      {
        event.preventDefault();
        $(".upload_container").addClass("loading");
        $(".form_container").hide(300)
        $(".result_container").show(300)
        sendData(this);
      });
}