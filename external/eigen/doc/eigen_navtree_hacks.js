
// generate a table of contents in the side-nav based on the h1/h2 tags of the current page.
function generate_autotoc() {
  var headers = $("h1, h2");
  if(headers.length > 1) {
    var toc = $("#side-nav").append('<div id="nav-toc" class="toc"><h3>Table of contents</h3></div>');
    toc = $("#nav-toc");
    var footer  = $("#nav-path");
    var footerHeight = footer.height();
    toc = toc.append('<ul></ul>');
    toc = toc.find('ul');
    var indices = new Array();
    indices[0] = 0;
    indices[1] = 0;

    var h1counts = $("h1").length;
    headers.each(function(i) {
      var current = $(this);
      var levelTag = current[0].tagName.charAt(1);
      if(h1counts==0)
        levelTag--;
      var cur_id = current.attr("id");

      indices[levelTag-1]+=1;  
      var prefix = indices[0];
      if (levelTag >1) {
        prefix+="."+indices[1];
      }
        
      // Uncomment to add number prefixes
      // current.html(prefix + "   " + current.html());
      for(var l = levelTag; l < 2; ++l){
          indices[l] = 0;
      }

      if(cur_id == undefined) {
        current.attr('id', 'title' + i);
        current.addClass('anchor');
        toc.append("<li class='level" + levelTag + "'><a id='link" + i + "' href='#title" +
                    i + "' title='" + current.prop("tagName") + "'>" + current.text() + "</a></li>");
      } else {
        toc.append("<li class='level" + levelTag + "'><a id='" + cur_id + "' href='#title" +
                    i + "' title='" + current.prop("tagName") + "'>" + current.text() + "</a></li>");
      }
    });
    resizeHeight();
  }
}


var global_navtree_object;

// Overloaded to remove links to sections/subsections
function getNode(o, po)
{
  po.childrenVisited = true;
  var l = po.childrenData.length-1;
  for (var i in po.childrenData) {
    var nodeData = po.childrenData[i];
    if((!nodeData[1]) ||  (nodeData[1].indexOf('#')==-1)) // <- we added this line
      po.children[i] = newNode(o, po, nodeData[0], nodeData[1], nodeData[2], i==l);
  }
}

// Overloaded to adjust the size of the navtree wrt the toc
function resizeHeight() 
{
  var header  = $("#top");
  var sidenav = $("#side-nav");
  var content = $("#doc-content");
  var navtree = $("#nav-tree");
  var footer  = $("#nav-path");
  var toc     = $("#nav-toc");

  var headerHeight = header.outerHeight();
  var footerHeight = footer.outerHeight();
  var tocHeight    = toc.height();
  var windowHeight = $(window).height() - headerHeight - footerHeight;
  content.css({height:windowHeight + "px"});
  navtree.css({height:(windowHeight-tocHeight) + "px"});
  sidenav.css({height:windowHeight + "px"});
}

// Overloaded to save the root node into global_navtree_object
function initNavTree(toroot,relpath)
{
  var o = new Object();
  global_navtree_object = o; // <- we added this line
  o.toroot = toroot;
  o.node = new Object();
  o.node.li = document.getElementById("nav-tree-contents");
  o.node.childrenData = NAVTREE;
  o.node.children = new Array();
  o.node.childrenUL = document.createElement("ul");
  o.node.getChildrenUL = function() { return o.node.childrenUL; };
  o.node.li.appendChild(o.node.childrenUL);
  o.node.depth = 0;
  o.node.relpath = relpath;
  o.node.expanded = false;
  o.node.isLast = true;
  o.node.plus_img = document.createElement("img");
  o.node.plus_img.src = relpath+"ftv2pnode.png";
  o.node.plus_img.width = 16;
  o.node.plus_img.height = 22;

  if (localStorageSupported()) {
    var navSync = $('#nav-sync');
    if (cachedLink()) {
      showSyncOff(navSync,relpath);
      navSync.removeClass('sync');
    } else {
      showSyncOn(navSync,relpath);
    }
    navSync.click(function(){ toggleSyncButton(relpath); });
  }

  navTo(o,toroot,window.location.hash,relpath);

  $(window).bind('hashchange', function(){
     if (window.location.hash && window.location.hash.length>1){
       var a;
       if ($(location).attr('hash')){
         var clslink=stripPath($(location).attr('pathname'))+':'+
                               $(location).attr('hash').substring(1);
         a=$('.item a[class$="'+clslink+'"]');
       }
       if (a==null || !$(a).parent().parent().hasClass('selected')){
         $('.item').removeClass('selected');
         $('.item').removeAttr('id');
       }
       var link=stripPath2($(location).attr('pathname'));
       navTo(o,link,$(location).attr('hash'),relpath);
     } else if (!animationInProgress) {
       $('#doc-content').scrollTop(0);
       $('.item').removeClass('selected');
       $('.item').removeAttr('id');
       navTo(o,toroot,window.location.hash,relpath);
     }
  })

  $(window).on("load", showRoot);
}

// return false if the the node has no children at all, or has only section/subsection children
function checkChildrenData(node) {
  if (!(typeof(node.childrenData)==='string')) {
    for (var i in node.childrenData) {
      var url = node.childrenData[i][1];
      if(url.indexOf("#")==-1)
        return true;
    }
    return false;
  }
  return (node.childrenData);
}

// Modified to:
// 1 - remove the root node 
// 2 - remove the section/subsection children
function createIndent(o,domNode,node,level)
{
  var level=-2; // <- we replaced level=-1 by level=-2
  var n = node;
  while (n.parentNode) { level++; n=n.parentNode; }
  if (checkChildrenData(node)) { // <- we modified this line to use checkChildrenData(node) instead of node.childrenData
    var imgNode = document.createElement("span");
    imgNode.className = 'arrow';
    imgNode.style.paddingLeft=(16*level).toString()+'px';
    imgNode.innerHTML=arrowRight;
    node.plus_img = imgNode;
    node.expandToggle = document.createElement("a");
    node.expandToggle.href = "javascript:void(0)";
    node.expandToggle.onclick = function() {
      if (node.expanded) {
        $(node.getChildrenUL()).slideUp("fast");
        node.plus_img.innerHTML=arrowRight;
        node.expanded = false;
      } else {
        expandNode(o, node, false, false);
      }
    }
    node.expandToggle.appendChild(imgNode);
    domNode.appendChild(node.expandToggle);
  } else {
    var span = document.createElement("span");
    span.className = 'arrow';
    span.style.width   = 16*(level+1)+'px';
    span.innerHTML = '&#160;';
    domNode.appendChild(span);
  }
}

// Overloaded to automatically expand the selected node
function selectAndHighlight(hash,n)
{
  var a;
  if (hash) {
    var link=stripPath($(location).attr('pathname'))+':'+hash.substring(1);
    a=$('.item a[class$="'+link+'"]');
  }
  if (a && a.length) {
    a.parent().parent().addClass('selected');
    a.parent().parent().attr('id','selected');
    highlightAnchor();
  } else if (n) {
    $(n.itemDiv).addClass('selected');
    $(n.itemDiv).attr('id','selected');
  }
  if ($('#nav-tree-contents .item:first').hasClass('selected')) {
    $('#nav-sync').css('top','30px');
  } else {
    $('#nav-sync').css('top','5px');
  }
  expandNode(global_navtree_object, n, true, true); // <- we added this line
  showRoot();
}


$(document).ready(function() {
  
  generate_autotoc();
  
  (function (){ // wait until the first "selected" element has been created
    try {
      
      // this line will triger an exception if there is no #selected element, i.e., before the tree structure is complete.
      document.getElementById("selected").className = "item selected";
      
      // ok, the default tree has been created, we can keep going...
      
      // expand the "Chapters" node
      if(window.location.href.indexOf('unsupported')==-1)
        expandNode(global_navtree_object, global_navtree_object.node.children[0].children[2], true, true);
      else
        expandNode(global_navtree_object, global_navtree_object.node.children[0].children[1], true, true);
      
      // Hide the root node "Eigen"
      $(document.getElementsByClassName('index.html')[0]).parent().parent().css({display:"none"});
      
    } catch (err) {
      setTimeout(arguments.callee, 10);
    }
  })();

  $(window).on("load", resizeHeight);
});

