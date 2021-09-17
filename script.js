$(document).ready(function(){main()});

function make_result_info(data){
  $(".info_container .chance").animate({num: data.result * 100 - 3}, {
    duration: 2000,
    step: function (num) {
      percent = (num + 3).toFixed(2)
      this.innerHTML = "A chance of COVID: " + percent + "%"
    }
  });
  if(data.data.defeat_square==3){
      $("body").append("<img src='luka.png' class='wheel'>");
      setTimeout(function(){
        $(".luka").animate({  textIndent: 0 }, {
            step: function(now,fx) {
              $(this).css('-webkit-transform','rotate('+now+'deg)'); 
            },
            duration:'slow'
        },'linear')}, 200
      );
  }
  text = "Defeats <b>" + data.data.defeat_square + "%</b> of lungs<br><br>"+
            "Left lung: " + data.data.left_defeat + " defeats<br>"+
            "Right lung: " + data.data.right_defeat + " defeats<br>"
  $(".info_container .res").html(text)
  
  stats_text = "Classification: " + data.data.stats.first_net + " s.<br>"+
                "Segmentation: " + data.data.stats.second_net + " s.<br>"+
                "Total: " + data.data.stats.all_time + " s."
  $(".info_container .stats").html(stats_text)
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
        is_covid = Boolean(parseInt(res.result.toFixed()))
        if(is_covid){
          $(".info_container .chance").addClass("bad_chance")
        }
        console.log(res)
        $(".result_image").attr("src", res.data.img_url[0]);
        $(".result1_image").attr("src", res.data.img_url[1]);
        $(".result2_image").attr("src", res.data.img_url[2]);
        $(".upload_container").addClass("upload_container_right")
        $(".result_image").show(300)
        $(".info_container").show(300);
        make_result_info(res, is_covid)
      } );
      XHR.addEventListener("error", function( event ) {
        alert("Error!");
        location.reload();
      } );
      XHR.open("POST", "http://localhost:5000/predict");
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