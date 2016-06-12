module.exports = (robot) ->

  robot.hear /((http|https).*)/i, (msg) ->

    image_url = escape(msg.match[1])
    server_url = "http://localhost:9000/api/hubot?url=" + image_url

    robot.http(server_url).get() (err, res, body) ->

      if err
        msg.send "Encountered an error :( #{err}"

      switch res.statusCode
        when 200
          response = switch body
            when "atypical"
              "This could potentially be dangerous. Please consult a doctor."
            when "common"
              "Everything is fine. There is nothing to worry about."
            when "melanoma"
              "This looks nasty. Please see a doctor."
	    else
	      ""

          msg.send(response)
        when 400
          msg.send("Ups. Something went wront with the mole server")
