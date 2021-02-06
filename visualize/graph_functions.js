d3.json("graph.json", function (error, graph) {

    var width = 1400,
        height = 1400;
    var color = d3.scale.category10();


    var force = d3.layout.force()
        .charge(-80)
        .linkDistance(20)
        .linkStrength(0.5)
        .gravity(0.2)
        .friction(0.4)
        .size([width, height]);


    var svg = d3.select("#d3-container").select("svg");
    if (svg.empty()) {
        svg = d3.select("#d3-container").append("svg")
            .attr("width", width)
            .attr("height", height);
    }


    defs = svg.append("defs");
    defs.append("marker")
        .attr({
            "id": "arrow",
            "viewBox": "0 -5 10 10",
            "refX": 30,
            "refY": 0,
            "markerWidth": 6,
            "markerHeight": 6,
            "orient": "auto"
        })
        .append("path")
        .attr("d", "M0,-5L10,0L0,5")
        .attr("class", "arrowHead");



    force.nodes(graph.nodes)
        .links(graph.links)
        .start();

    var drag = force.drag()
        .on("dragend", dragend);

    min_weight = Math.min.apply(null,
        Object.keys(graph.links).map(function(e) {return graph.links[e].weight}));
    max_weight = Math.max.apply(null,
        Object.keys(graph.links).map(function(e) {return graph.links[e].weight}));
    var clr_range = d3.scale.linear()
        .domain([min_weight, max_weight])
        .range(["red", "black"]);


    var link = svg.selectAll(".link")
        .data(graph.links)
        .enter().append("line")
        .style('stroke', function (l) {
            return clr_range(l.weight)

        })
        .attr('id', function (l) {
            return "e" + l.source.index + "_" + l.target.index;
        })
        .attr("class", "link")
        .attr("marker-end", "url(#arrow)");


    var node = svg.selectAll(".node")
        .data(graph.nodes)
        .enter().append("circle")
        .attr("class", "node")
        .style("fill", function (d) {
            return color(d.color);
        })
        .style("r", function (d) {
            if (d.is_exemplar == true){
                return 10
            }else{
                return 5
            }
        })
        .attr('neighbours', function (d) {
            return d.neighbours;
        })
        .call(drag);

    //Toggle stores whether the highlighting is on
    var toggle = 0;
    var linkedByIndex = {};
    for (var i = 0; i < graph.nodes.length; i++) {
        linkedByIndex[i + "," + i] = 1;
    }
    graph.links.forEach(function (d) {
        linkedByIndex[d.source.index + "," + d.target.index] = 1;
    });


    // The label each node its node number from the networkx graph.
    node.append("title")
        .text(function (d) {
            return "Node: " + d.id + "\n" + "Degree: " + d.degree + "\n" + "Name: " + d.name;
        });


    //////////////////////////////////////////
    node_neighbours = {};
    node.each(function (d1) {
        node_neighbours[d1.id] = [];
        node.each(function (d2) {
            if (neighboring(d1, d2) | neighboring(d2, d1)) {
                node_neighbours[d1.id].push(d2.id)
            }
        });
    });

    node_edges = {};
    node.each(function (d) {
        node_edges[d.id] = [];
        link.each(function (l) {
            if (d.index == l.source.index | d.index == l.target.index) {
                node_edges[d.id].push(l)
            }
        });

    });
    //////////////////////////////////////////

    node.on('mouseover', connectedNodes);
    node.on('mouseout', function () {
        node.style("opacity", 1);
        link.style("opacity", 1);
        link.style("stroke-width", 1);
        document.getElementById("imgCanvas").style.display= 'none';

    });
    node.on('dblclick', function (d1) {
        neighbours_names = [];
        node.each(function (d2) {
            if (neighboring(d1, d2) | neighboring(d2, d1)) {
                neighbours_names.push(d2.name)
            }
        });

        // console.log(neighbours_names);
        window.open(
            "http://www.mercateo.com/p/compare/?ID=" + d1.name +
            "&ID=" + neighbours_names.join('&ID=') + "&ViewName=allLive",
            '_blank'
        )
    });



    force.on("tick", function () {
        link.attr("x1", function (d) {
            return d.source.x;
        })
            .attr("y1", function (d) {
                return d.source.y;
            })
            .attr("x2", function (d) {
                return d.target.x;
            })
            .attr("y2", function (d) {
                return d.target.y;
            });

        node.attr("cx", function (d) {
            return d.x;
        })
            .attr("cy", function (d) {
                return d.y;
            });


    });


    function dragend(d) {
        // d3.select(this).classed("fixed", d.fixed = true);
        if (d.fixed == false) {
            d3.select(this).classed("fixed", d.fixed = true);
        } else {
            d3.select(this).classed("fixed", d.fixed = false);
        }

    }

    function connectedNodes() {
        //Reduce the opacity of all but the neighbouring nodes to 0.3.
        var d = d3.select(this).node().__data__;


        node.style("opacity", function (o) {
            return neighboring(d, o) | neighboring(o, d) ? 1 : 0.3;
        });
        //Reduce the opacity of all but the neighbouring edges to 0.8.
        link.style("opacity", function (o) {
            return d.index == o.source.index | d.index == o.target.index ? 1 : 0.3;
        });
        //Reset the toggle.
        toggle = 1;

        /////////////////////////////////////////////////
        if (d.extra != 'undefined' && d.extra.img != undefined) {
            var c = document.getElementById("imgCanvas");
            c.style.display = 'block';
            var ctx = c.getContext("2d");
            var width = 80,
                height = 80;

            buffer = new Uint8ClampedArray(d.extra.img.length * d.extra.img[0].length * 4); // have enough bytes
            for (var y = 0; y < d.extra.img.length; y++) {
                for (var x = 0; x < d.extra.img[0].length; x++) {
                    var pos = (y * d.extra.img.length + x) * 4; // position in buffer based on x and y
                    buffer[pos + 0] = d.extra.img[y][x] * 255;           // some R value [0, 255]
                    buffer[pos + 1] = d.extra.img[y][x] * 255;           // some G value
                    buffer[pos + 2] = d.extra.img[y][x] * 255;           // some B value
                    buffer[pos + 3] = 255;           // set alpha channel
                }
            }
            c.width = width;
            c.height = height;
            var idata = ctx.createImageData(d.extra.img.length, d.extra.img[0].length);
            idata.data.set(buffer);
            ctx.putImageData(idata, 0, 0);
            scale_x = width / d.extra.img.length;
            scale_y = height / d.extra.img[0].length;
            ctx.drawImage(c, 0, 0, scale_x * c.width, scale_y * c.height);
            c.style.left = d3.event.pageX + 10 + 'px';
            c.style.top = d3.event.pageY - 150 + 'px';

        }


    }

    //Looks up whether a pair of nodes are neighbours.
    function neighboring(a, b) {
        return linkedByIndex[a.index + "," + b.index];
    }


});



