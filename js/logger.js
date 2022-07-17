export default function log(text, classes='debug') {
    document.querySelector('div#logs').innerHTML += '<div class="log '+classes+'">'+text+'</div>';
    document.querySelector('.log:nth-last-child(1)').scrollIntoView()
}
