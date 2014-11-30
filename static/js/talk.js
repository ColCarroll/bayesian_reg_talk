/**
 * Created by colinc on 9/7/14.
 */
var randomProperty = function (obj) {
    var keys = Object.keys(obj);
    var key = keys[ keys.length * Math.random() << 0];
    return [key, obj[key]];
};

var talk = angular.module("linregTalk", []);

talk.controller('TalkCtrl',
    function TalkCtrl($scope, $http){
        $scope.linX = d3.range(-1, 1, 0.01);
        $scope.restoreDefaults = function(){
            $scope.parameters = {
                "noise" : 0.1,
                "trainingPoints": 100,
                "minX": -1,
                "maxX": 1,
                "generating_functions": {
                    "1 - 0.2 x": function(x){return 1 - 0.2 * x},
                    "x^2": function(x){return x * x},
                    "\\sin{(5x)}": function(x){return Math.sin(5 * x)},
                    "e^x": function(x){return Math.exp(x)},
                    "\\frac{1}{1 + e^{-9x}}": function(x){ return 1.0 / (1 + Math.exp(-9 * x))},
                    "\\sin{\\frac{1}{x}}": function(x) { return Math.sin(1.0 / x)}
                }
            };
            $scope.generating_function = $scope.parameters.generating_functions["x^2"];
            $scope.generating_function_name = "x^2";
        };
        $scope.restoreDefaults();

        var random = d3.random.normal(0, 1);

        $scope.getFunc = function(func){
            var y = _.map($scope.linX, func);
            return _.map(
                _.zip($scope.linX, y),
                function (coords) {
                    return {"x": coords[0], "y": coords[1]}
                });
        };
        $scope.sigmoid = $scope.getFunc(function(j){return 1.0 / (1 + Math.exp(-5 * j));});
        $scope.polynomial = $scope.getFunc(function(j){return Math.pow(j, 2);});
        $scope.gaussian = $scope.getFunc(function(j){return Math.exp(-Math.pow(j, 2) / 0.25);});

        $scope.cycleData = function(){
            $scope.parameters.trainingPoints = 1 + Math.abs(Math.round(random() * 200));
            var gen_func = randomProperty($scope.parameters.generating_functions);
            $scope.generating_function_name = gen_func[0];
            $scope.generating_function = gen_func[1];
            $scope.parameters.noise = 0.3 * Math.abs(random());
            $scope.updateX()
        };

        $scope.updateMathJaxStr = function(){
            $scope.latexGenFunc = "y(x) = " + $scope.generating_function_name + " + \\mathbf{N}(0, " + $scope.parameters.noise.toFixed(2) + ")";

        };

        $scope.updateX = function(){
            var parms = $scope.parameters;
            $scope.x = _.map(_.range(parms.trainingPoints), function(){return (parms.maxX - parms.minX) * Math.random() + parms.minX });
            $scope.updateNoise();
        };

        $scope.updateNoise = function(){
            $scope.noise = _.map($scope.x, function(){ return random()});
            $scope.updateMathJaxStr();
            $scope.updateData();
        };

        $scope.updateData = function(){
            var y = _.map($scope.x, $scope.generating_function);
            var yNoise = _.map($scope.x, function(x, i) { return $scope.generating_function(x) + $scope.parameters.noise * $scope.noise[i]});
            $scope.data = _.map(
                _.zip($scope.x, y, yNoise),
                function (coords) {
                    return {"x": coords[0], "y": coords[1], "yNoise": coords[2]}
                });
        };
        $scope.updateX();

        $scope.mParms = {
            "training_points": 30,
            "testing_points": 100,
            "num_basis": 3,
            "func_no": 1,
            "noise": 0.05
        };

        $scope.cycleModel = function(){
            $scope.mParms.training_points = 1 + Math.abs(Math.round(random() * 50));
            $scope.updateModel("training_points", $scope.mParms.training_points);
            $scope.mParms.func_no = Math.round(5 * Math.random());
            $scope.updateModel("func_no", $scope.mParms.func_no);
            $scope.mParms.noise = 0.3 * Math.abs(random());
            $scope.updateModel("noise", $scope.mParms.noise);
        };

        $scope.getModel = function(){
            $http({
                method: 'GET',
                url: "/data"
            }).success(function(data){
                $scope.testModel = data.test;
                $scope.trainModel = data.train;
                $scope.plotModel = data.plot;
                $scope.errors = data.errors;
            });
        };

        $scope.getModel();

        $scope.updateModel = function(trait, newVal){
            $http({
                method: "GET",
                url: "/update?" + trait + "=" + newVal
            }).success(function(data){
                $scope.testModel = data.test;
                $scope.trainModel = data.train;
                $scope.plotModel = data.plot;
                $scope.errors = data.errors;
            })
        }
    }
);

talk.directive('talkSlides', [
        function(){
            return {
                restrict: 'A',
                link: function(scope, element, attrs){
                    element.addClass('slides');

                    if(!window.Reveal){
                        return;
                    }
                    window.Reveal.initialize({
                        controls: true,
                        progress: true,
                        history: true,
                        center: true,
                        backgroundTransition: "slide",
                        parallaxBackgroundImage: "/rice.png",
                        parallaxBackgroundSize: "4813px 1325px",

                        transition: 'linear'
                    });
                }
            }
        }
    ]
)
    .directive("mathjaxBind", function() {
        return {
            restrict: "A",
            controller: ["$scope", "$element", "$attrs",
                function($scope, $element, $attrs) {
                    $scope.$watch($attrs.mathjaxBind, function(texExpression) {
                        var texScript = angular.element("<script type='math/tex'>")
                            .html(texExpression ? texExpression :  "");
                        $element.html("");
                        $element.append(texScript);
                        MathJax.Hub.Queue(["Reprocess", MathJax.Hub, $element[0]]);
                    });
                }]
        }
    }
);

talk.directive("plotPoints", function() {
        // constants
        var margin = {top: 20, right: 20, bottom: 30, left: 40},
            height = 500,
            width=760;

        return {
            restrict: "EA",
            scope: {
                pointData: '='
            },
            link: function (scope, element, attrs) {
                var svg = d3.select(element[0]).append("svg")
                    .style("width", width + margin.left + margin.right)
                    .style("height", height + margin.top + margin.bottom);

                var xValue = function (d) {return d.x;},
                    xScale = d3.scale.linear().range([0, width]),
                    xMap = function (d) {return xScale(xValue(d));},
                    xAxis = d3.svg.axis().scale(xScale).orient("bottom");

                var yValue = function (d) {return d.yNoise;},
                    yScale = d3.scale.linear().range([height, 0]),
                    yMap = function (d) {return yScale(yValue(d));},
                    yAxis = d3.svg.axis().scale(yScale).orient("right");

                // x-axis
                svg.append("g")
                    .attr("class", "x axis")
                    .attr("transform", "translate(0," + height + ")")
                    .call(xAxis)
                    .append("text")
                    .attr("class", "label")
                    .attr("x", width - 5)
                    .attr("y", -2);

                // y-axis
                svg.append("g")
                    .attr("class", "y axis")
                    .attr("transform", "translate(" + width + ", 0)")
                    .call(yAxis)
                    .append("text")
                    .attr("class", "label")
                    .attr("transform", "rotate(-90)")
                    .attr("y", 6)
                    .attr("dy", "-1.2em");

                svg.attr("height", height + margin.top + margin.bottom)
                    .append("g")
                    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

                yScale.domain([d3.min(scope.pointData, yValue) - 0.2, d3.max(scope.pointData, yValue) + 0.2]);

                svg.selectAll("g.y.axis")
                    .call(yAxis);

                svg.selectAll("g.x.axis")
                    .call(xAxis);

                var renderPoints = function (data) {
                    if (!data || isNaN(data.length)) return;
                    yScale.domain([d3.min(data, yValue) - 0.2, d3.max(data, yValue) + 0.2]);

                    var points =  svg.selectAll(".dot")
                        .data(data, function(d, i){return i;});

                    points.enter()
                        .append("circle")
                        .attr("class", "dot")
                        .attr("opacity", 0.8)
                        .attr("r", 5)
                        .attr("cx", xMap)
                        .attr("cy", function(d){ return yScale(d.y)});

                    points.transition()
                        .duration(500)
                        .attr("cy", function(d){ return yScale(d.y)})
                        .attr("cx", xMap)
                        .transition()
                        .duration(500)
                        .attr("cy", yMap)
                        .attr("cx", xMap);

                    points.exit()
                        .remove();
                };
                renderPoints(scope.pointData);
                scope.$watch("pointData", renderPoints);
            }
        }
    }
);

talk.directive("plotLine", function(){
        // constants
        var margin = {top: 20, right: 20, bottom: 30, left: 40},
            height = 500,
            width=760;

        return {
            restrict: "EA",
            scope: {
                pointData: '='
            },
            link: function (scope, element, attrs) {
                var svg = d3.select(element[0]).append("svg")
                    .style("width", width + margin.left + margin.right)
                    .style("height", height + margin.top + margin.bottom);

                var xValue = function (d) {return d[attrs.xkey];},
                    xScale = d3.scale.linear().range([0, width]).domain([-1.1, 1.1]),
                    xMap = function (d) {return xScale(xValue(d));},
                    xAxis = d3.svg.axis().scale(xScale).orient("bottom");

                var yValue = function (d) {return d[attrs.ykey];},
                    yScale = d3.scale.linear().range([height, 0]),
                    yMap = function (d) {return yScale(yValue(d));},
                    yAxis = d3.svg.axis().scale(yScale).orient("right");

                // x-axis
                svg.append("g")
                    .attr("class", "x axis")
                    .attr("transform", "translate(0," + height + ")")
                    .call(xAxis)
                    .append("text")
                    .attr("class", "label")
                    .attr("x", width - 5)
                    .attr("y", -2);

                // y-axis
                svg.append("g")
                    .attr("class", "y axis")
                    .attr("transform", "translate(" + width + ", 0)")
                    .call(yAxis)
                    .append("text")
                    .attr("class", "label")
                    .attr("transform", "rotate(-90)")
                    .attr("y", 6)
                    .attr("dy", "-1.2em");

                svg.attr("height", height + margin.top + margin.bottom)
                    .append("g")
                    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

                yScale.domain([d3.min(scope.pointData, yValue) - 0.2, d3.max(scope.pointData, yValue) + 0.2]);

                svg.selectAll("g.y.axis")
                    .call(yAxis);

                svg.selectAll("g.x.axis")
                    .call(xAxis);

                var line = d3.svg.line()
                    .x(xMap)
                    .y(yMap);

                svg.append("path")
                    .datum(scope.pointData)
                    .attr("class", "line")
                    .attr("d", line);
            }
        }
    }
);

talk.directive("plotModel", function(){
        // constants
        var margin = {top: 20, right: 20, bottom: 30, left: 40};

        return {
            restrict: "EA",
            scope: {
                modelPoint: "=",
                modelLine: "="
            },
            link: function (scope, element, attrs) {
                var width = parseInt(attrs.width),
                    height = parseInt(attrs.height);

                var svg = d3.select(element[0]).append("svg")
                    .style("width", width + margin.left + margin.right)
                    .style("height", height + margin.top + margin.bottom);

                var xScale = d3.scale.linear().range([0, width]).domain([-1.2, 1.2]),
                    xValue = function(d){return d[0];},
                    xMap = function(d){return xScale(xValue(d));},
                    xAxis = d3.svg.axis().scale(xScale).orient("bottom");

                var yScale = d3.scale.linear().range([height, 0]).domain([-0.1, 1.1]),
                    yValue = function(d){return d[1];},
                    yMap = function (d) {return yScale(yValue(d));},
                    yAxis = d3.svg.axis().scale(yScale).orient("right");

                // x-axis
                svg.append("g")
                    .attr("class", "x axis")
                    .attr("transform", "translate(0," + height + ")")
                    .call(xAxis)
                    .append("text")
                    .attr("class", "label")
                    .attr("x", width - 5)
                    .attr("y", -2);

                // y-axis
                svg.append("g")
                    .attr("class", "y axis")
                    .attr("transform", "translate(" + width + ", 0)")
                    .call(yAxis)
                    .append("text")
                    .attr("class", "label")
                    .attr("transform", "rotate(-90)")
                    .attr("y", 6)
                    .attr("dy", "-1.2em");

                svg.attr("height", height + margin.top + margin.bottom)
                    .append("g")
                    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

                scope.$watch("modelPoint", function(d){return renderModel()});
                scope.$watch("modelLine", function(d){return renderModel()});

                var line = d3.svg.line()
                    .x(xMap)
                    .y(yMap);

                var renderModel = function() {
                    if (!scope.modelLine || isNaN(scope.modelLine.length)) return;
                    yScale.domain([
                            d3.min(scope.modelLine.concat(scope.modelPoint), yValue) - 0.2,
                            d3.max(scope.modelLine.concat(scope.modelPoint), yValue) + 0.2]);
                    xScale.domain([
                            d3.min(scope.modelLine.concat(scope.modelPoint), xValue) - 0.2,
                            d3.max(scope.modelLine.concat(scope.modelPoint), xValue) + 0.2]);

                    var points =  svg.selectAll(".dot")
                        .data(scope.modelPoint);

                    points.enter()
                        .append("circle")
                        .attr("class", "dot")
                        .attr("opacity", 0.8)
                        .attr("r", 5)
                        .attr("cx", xMap)
                        .attr("cy", yMap);

                    points.transition()
                        .transition()
                        .duration(500)
                        .attr("cy", yMap)
                        .attr("cx", xMap);

                    points.exit()
                        .remove();

                    var path =  svg.selectAll(".line")
                        .data([scope.modelLine]);

                    path.enter().append("path")
                        .attr("class", "line")
                        .attr("d", line);

                    path.transition()
                        .duration(500)
                        .attr("class", "line")
                        .attr("d", line);

                    svg.selectAll("g.y.axis")
                        .call(yAxis);

                    svg.selectAll("g.x.axis")
                        .call(xAxis);
                };
            }
        }
    }
);


