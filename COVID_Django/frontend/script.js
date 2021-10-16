$(document).ready(function () { main() });
var imgs = [];
var is_dicom = false;
var img_el;
var img_overlay_el;

function nextImage() {
  img_id = img_el.attr("id")
  next = parseInt(img_id) + 1
  setImageSrc(next)
}
function prevImage() {
  img_id = img_el.attr("id")
  next = parseInt(img_id) - 1
  setImageSrc(next)
}
function setImageSrc(id) {
  if (id < 0) {
    id = imgs[0].length-1;
  }
  if (id > imgs[0].length-1) {
    id = 0;
  }
  img_el.attr('id', id)
  img_el.attr('src', imgs[0][id])
  img_overlay_el.attr('src', imgs[1][id])
  $(".controls > #lungs").attr('value', id)
}
function lungsOpacity(value) {
  if(is_dicom){
    return setImageSrc(value)
  }
  $(".result1_image").css("opacity", value)
}
function defeatsOpacity(value) {
  if(is_dicom){
    return img_overlay_el.css("opacity", value)
  }
  $(".result2_image").css("opacity", value)
}

function make_result_info(data) {
  if (is_dicom) {
    text = "Left lung: " + data.data.left_affection_percent + "% (" + data.data.left_defeats_volume + "sm3) affected<br>" +
      "Right lung: " + data.data.right_affection_percent + "% (" + data.data.right_defeats_volume + "sm3) affected<br>"
    $(".info_container .res").html(text)
    $(".info_container .stats").html("Total: " + data.data.stats.all_time + " s.")
    $("#text_range_lungs").text("Scroll images")
  } else {
    text = "Affected <b>" + data.data.affections_square + "%</b> of lungs<br><br>" +
      "Left lung: " + data.data.left_affections + " affections<br>" +
      "Right lung: " + data.data.right_affections + " affections<br>"
    $(".info_container .res").html(text)
    $(".info_container .stats").html("Total: " + data.data.stats.all_time + " s.")
  }
}

function main() {
  $("#files").change(function () {
    filename = this.files[0].name
    console.log(filename);
  });
  img_el = $(".result_image")
  img_overlay_el = $(".result1_image")
  $(".refresh_page").click(function () { location.reload() })
  $(".form_image").change(function (event) {
    event.preventDefault();
    $(".upload_container").addClass("loading");
    $(".form_container").hide(300)
    $(".result_container").show(300)
    sendData(this);
  });
  $(".form_dicom").change(function (event) {
    event.preventDefault();
    $(".upload_container").addClass("loading");
    $(".form_container").hide(300)
    $(".result_container").show(300)
    sendDicomData(this);
  });
  document.body.addEventListener("mousewheel", (event) => {
    if (event.deltaY > 0) {
      prevImage()
    } else {
      nextImage()
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "ArrowUp") {
      prevImage()
    } else if (event.key === "ArrowDown") {
      nextImage()
    }
  });
  $(".btn_download").click(function(){$("#form_download").submit()})
}
function sendData(form) {
  const XHR = new XMLHttpRequest();
  const FD = new FormData(form);
  XHR.addEventListener("load", function (event) {
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
  });
  XHR.addEventListener("error", function (event) {
    alert("Error!");
    location.reload();
  });
  XHR.open("POST", "//" + document.domain + ":5000/api/predict");
  XHR.send(FD);
}
function sendDicomData(form) {
  const XHR = new XMLHttpRequest();
  const FD = new FormData(form);
  XHR.addEventListener("load", function (event) {
    is_dicom = true;
    $(".upload_container").removeClass("loading");
    res = event.target.responseText;
    res = $.parseJSON(res)
    console.log(res)
    $(".result2_image").remove();
    $(".result1_image").show();
    $(".upload_container").addClass("upload_container_right")
    $(".result_image").show(300)
    $(".info_container").show(300);
    make_result_info(res, true);
    imgs = res.data.img_urls;
    archive_url = res.data.archive;
    $("#form_download").attr("action", archive_url);
    $(".btn_download").show();
    $(".controls > #lungs").attr('max', imgs[0].length-1)
    $(".controls > #lungs").attr('step', 1)
    $(".controls > #lungs").attr('value', 0)
    setImageSrc(0)
  });
  XHR.addEventListener("error", function (event) {
    alert("Error!");
    location.reload();
  });
  XHR.open("POST", "//" + document.domain + ":5000/api/predict_dicom");
  XHR.send(FD);
}