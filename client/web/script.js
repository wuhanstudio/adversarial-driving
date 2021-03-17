// Driving Data from the Server
var steer_data = [];
var adv_data = [];

// Connect to the WebSocket Server
socket = io.connect('http://localhost:4567');

socket.on('connect', function () {
    console.log('Client has connected to the server!');
    attack(0, $("input[name=flexRadioDefault]:checked").val());
});

socket.on('disconnect', function () {
    console.log('The client has disconnected!');
    $("#customSwitchActivate").prop("checked", false);
});

// Log messages
socket.on('info', function (data) {
    console.log(data);
});

// Receive the original image
socket.on('update', function (data) {
    $('#origin').attr("src", "data:image/png;base64," + data.data);
});

// Receive the input image
socket.on('input', function (data) {
    $('#input').attr("src", "data:image/png;base64," + data.data);
});

// Receive the perturbation
socket.on('diff', function (data) {
    // console.log('Received a message from the server!',data);
    $('#diff').attr("src", "data:image/png;base64," + data.data);
});

// Receive the adversarial image
socket.on('adv', function (data) {
    $('#adv').attr("src", "data:image/png;base64," + data.data);
});

// Receive Training result for UAPr
socket.on('unir_train', function (data) {
    $("#train_res").text("Train: " + parseFloat(data.absolute).toFixed(2) + ' ' + parseFloat(data.percentage).toFixed(2) + "%");
});

// Sends a message to the server via sockets
function send(message) {
    socket.emit('attack', message);
};

function attack(isAttack, type) {
    message = {};
    message.attack = isAttack;
    message.type = type;

    console.log(message);
    send(message);
}

// Chart Options
var options = {
    series: [
        {
            name: "Without attack",
            data: steer_data.slice()
        },
        {
            name: "With attack",
            data: adv_data.slice()
        },
    ],
    chart: {
        id: 'realtime',
        height: 350,
        type: 'line',
        animations: {
            enabled: true,
            easing: 'linear',
            dynamicAnimation: {
                speed: 1000
            }
        },
        toolbar: {
            show: false
        },
        zoom: {
            enabled: false
        }
    },
    colors: ['#77B6EA', '#545454'],
    dataLabels: {
        enabled: false
    },
    stroke: {
        curve: 'smooth'
    },
    title: {
        text: 'Steering Angle',
        align: 'left',
        style: {
            fontSize: '20px'
        }
    },
    markers: {
        size: 0
    },
    xaxis: {
        type: 'line',
        labels: {
            show: false
        }
        // range: XAXISRANGE,
    },
    yaxis: {
        min: -100,
        max: 100,
        labels: {
            style: {
                fontSize: '18px',
            }
        },
        decimalsInFloat: 1,
        tickAmount: 10
    },
    legend: {
        fontSize: '22px',
        position: 'top',
        horizontalAlign: 'right',
        floating: true,
        offsetY: -25,
        offsetX: -5
    }
};

// Attack Deactivated
function resume() {
    $('#diff').attr("src", "./hold.png");
    $('#adv').attr("src", "./hold.png");
}

$(document).ready(function () {
    $('#uni_train_btn').hide();

    // Select different attacks
    $("input[name=flexRadioDefault]").change(function () {
        attack(0, this.value);
        if (this.value === "unir_no_left" || this.value === "unir_no_right") {
            $('#uni_train_btn').show();
        }
        else {
            $('#uni_train_btn').hide();
        }
        $("#customSwitchActivate").prop("checked", false);
        $("#origin").css("border-style", "none");
    });

    // Activate different attacks
    $('#customSwitchActivate').change(function () {
        if ($(this).prop('checked')) {
            attack(1, $("input[name=flexRadioDefault]:checked").val());
            $("#origin").css("border-style", "solid");
            $("#origin").css("border-color", "coral");
            $("#customSwitchTrain").prop("checked", false);
        }
        else {
            attack(0, $("input[name=flexRadioDefault]:checked").val());
            $("#origin").css("border-style", "none");
            resume();
        }
    })

    // Activate traning / learning for Universal Adversarial Perturbation
    $('#customSwitchTrain').change(function () {
        if ($(this).prop('checked')) {
            attack(1, $("input[name=flexRadioDefault]:checked").val() + '_train');
            $("#origin").css("border-style", "solid");
            $("#origin").css("border-color", "coral");
            $("#customSwitchActivate").prop("checked", false);
        }
        else {
            attack(0, $("input[name=flexRadioDefault]:checked").val() + '_train');
            $("#origin").css("border-style", "none")
        }
    })

    var chart = new ApexCharts(document.querySelector("#chart"), options);
    chart.render();

    // Receive Attack results
    socket.on('res', function (data) {
        $("#attack_res").text("Attack: From " + parseFloat(data.original).toFixed(2) + ' to ' + parseFloat(data.result).toFixed(2) + ', ' + parseFloat(data.percentage).toFixed(2) + "%");

        steer_data.push(parseFloat(data.original) * 100);
        adv_data.push(parseFloat(data.result) * 100);
        if (steer_data.length > 50) {
            steer_data.shift();
        }
        if (adv_data.length > 50) {
            adv_data.shift();
        }

        chart.updateSeries([{ data: steer_data }, { data: adv_data }]);
    });
});
