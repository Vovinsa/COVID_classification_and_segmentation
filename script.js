$(document).ready(function(){main()});

function make_result_info(data){
  $(".info_container .chance").animate({num: data.result * 100 - 3}, {
    duration: 2000,
    step: function (num) {
      percent = (num + 3).toFixed(2)
      this.innerHTML = "Шанс ковида: " + percent + "%"
    }
  });
  text = "Поражено <b>" + data.data.defeat_sqare + "%</b> легких<br><br>"+
            "Левое легкое: " + data.data.left_defeat + " поражений<br>"+
            "Правое легкое: " + data.data.right_defeat + " поражений<br>"
  $(".info_container .res").html(text)
  
  stats_text = "Классификация: " + data.data.stats.first_net + " сек.<br>"+
                "Детекция: " + data.data.stats.second_net + " сек.<br>"+
                "Всего: " + data.data.stats.all_time + " сек."
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
        $(".result_image").attr("src", res.data.img_url);
        $(".upload_container").addClass("upload_container_right")
        $(".result_image").show(300)
        $(".info_container").show(300);
        make_result_info(res, is_covid)
      } );
      XHR.addEventListener("error", function( event ) {
        alert("Произошла ошибка!");
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
