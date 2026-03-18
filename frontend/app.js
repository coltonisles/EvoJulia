

let genBtn = document.getElementById('generateBtn')
let imgIn = document.getElementById('imgInput')
let targetOutput = document.getElementById('targetDisplay')
let fractalOutput = document.getElementById('fractalDisplay')

genBtn.addEventListener('click', async () => {
    let formData = new FormData();
    if (imgIn.files.length > 0) {
        targetOutput.src = URL.createObjectURL(imgIn.files[0])
        formData.append('img', imgIn.files[0])
    }else{
        alert('Please select an image')
    }

})